import math

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .layers import LoRALayer 
from typing import Optional, List
import scipy.stats as st
import networkx as nx
from joblib import Parallel, delayed

def remove_min_edge_in_cycles(graph):
    try:
        while True:
            cycle = nx.find_cycle(graph)

            min_weight = float('inf')
            min_edge = None

            for edge in cycle:
                u, v = edge[0], edge[1]
                weight = graph[u][v]['weight']
                if weight < min_weight:
                    min_weight = weight
                    min_edge = (u, v)

            if min_edge:
                graph.remove_edge(*min_edge)

    except nx.NetworkXNoCycle:
        # 当图中没有更多环时，会抛出 NetworkXNoCycle 异常
        pass

    return graph

def compute_weights(graph):
    """
    computer the influence domain of dependency graph
    """
    n = len(graph)
    weight = [0 for i in range(n)]
    outdegree = [0 for i in range(n)]
    for i in range(n):
        outdegree[i] = sum(1 for j in range(n) if graph[i][j]!=0 )


    def importance(node):
        if weight[node]!= 0:
            return weight[node]
        elif outdegree[node]==0:
            return 1
        for j in range(n):
            if graph[node][j]!=0:
                weight[node] += graph[node][j] * importance(j)
        return weight[node]


    for i in range(n):
        if outdegree[i] != 0:
            weight[i] = importance(i)

    return weight


class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            ) 
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                ) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class RankAllocator(object):
    """
    The RankAllocator for SalientLoRA Model.

    Args:
        model: the model that we apply salientlora to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter.

        `average_initial_rank`:  The initial rank of each incremental matrix.
        `average_target_rank`: The average target rank of each incremental matrix.
        `initial_warmup`:  The number of steps to warm up the training before rank allocation.
        `allocation_step`: The number of steps during the rank allocation phase.
        `initial_time_window `: The initial window size of time-series.
        `final_time_window`: The final window size of time-series.
        `beta`: The correlation threshold.
        `gamma`: The slope threshold for dependency calculation.
        `lambda_para`: The hyperparameter controlling the degree of contribution in salience measurement.
    """
    def __init__(
        self, model, 
        average_initial_rank:int,
        average_target_rank:int,
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        tb_writter=None,
        tb_writter_loginterval:int=500, 
    ):
        self.ave_target_rank = average_target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = average_initial_rank
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.regu_loss = []
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()   # 总秩数是通过lora_r和所有的矩阵来算的，目标秩数是通过平均目标秩数来算的
        self.global_step = 0
        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval
        self.mask_weight = {}
        self.weight_t = []
        self.T_cur = 0
        self.T_left = 10



        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 

    def adaptive_time_window(self, step:int):
        # the adjustment of adaptive_time-series window
        mask_ind = False 
        target_rank = self.target_rank
        initial_rank = self.lora_init_rank * len(self.mask_weight)
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step
        self.global_step = step
        T_finish = 200   # 超参数，先暂时写在这里，应该要放到代码运行前指定。
        T_init = 10
        if step <= initial_warmup:
            # Initial warmup  第一阶段，
            curr_rank = self.total_rank 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            # Final fine-tuning
            curr_rank = self.target_rank 
            # Fix the rank pattern by 
            # always masking the same unimportant singluar values
            mask_ind = False
        else: 
            # Budget decreasing
            coeff = (step - initial_warmup) /(total_step - final_warmup)
            T = round(T_finish + (T_init - T_finish)*(1 - coeff)**3)    # 当前时间窗口
            self.T_left -= 1
            curr_rank = int(initial_rank - (T / T_finish) * (initial_rank - target_rank))
            if self.T_left == 0:
                mask_ind = True
                self.T_cur = T
                self.T_left = self.T_cur

        return curr_rank, mask_ind 



    def record_update(self, model):    # 记录并更新当前所有奇异值的权重
        """
        记录当前步数下奇异值的权重
        # 每次更新完参数后，进行参数mask，以保证参数被mask后不可恢复(梯度无法传播)。
        # p*mask
        :param model:
        :param global_step:
        :return:
        """
        weight = {}
        for n,p in model.named_parameters():
            if "lora_E" in n:
                if n not in self.mask_weight:
                    self.mask_weight[n] = torch.ones_like(p)
                with torch.no_grad():
                    p.data = p.data * self.mask_weight[n]
                weight[n] = p.data.detach().clone()
                # weight[n] = p.data
        if self.global_step <= self.total_step - self.final_warmup:
            self.weight_t.append(weight)



    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt


    def del_tensor_0_cloumn(Cs):
        idx = torch.where(torch.all(Cs[..., :] == 0, axis=0))[0]
        all = torch.arange(Cs.shape[1])
        for i in range(len(idx)):
            all = all[torch.arange(all.size(0)) != idx[i] - i]
        Cs = torch.index_select(Cs, 1, all)
        return Cs


    def mask_to_target_rank(self, model, curr_rank, Print_rank, root_output_dir):
        """
        只考虑重要性奇异值的参数大小
        :param model:
        :param curr_rank:
        :param Print_rank:
        :return:
        """
        is_dict = {}
        all_is = []
        # Calculate the importance score for each sub matrix
        # 正交化损失为：事件窗口内所有损失总和除以单个损失的值
        regu_weight = sum(self.regu_loss)/torch.Tensor(self.regu_loss).to(sum(self.regu_loss).device)
        min_vals = torch.min(regu_weight)
        max_vals = torch.max(regu_weight)
        regu_weight = (regu_weight - min_vals) / (max_vals - min_vals)

        tmp = []        # 将所有时间步内的奇异值都转化为一个张量矩阵中，维度为(self.total_rank, self.mask_interval). 每一列是一个时间步下的所有奇异值。按顺序排列
        # for i in range(self.mask_interval): # 修改
        for i in range(len(self.weight_t)):
            tmp.append(torch.cat([v for k,v in self.weight_t[i].items()]))
        all_weight = torch.cat(tmp,1)

        # 把权重等于0的index给去掉。  因为现在可以用并行算法，就不需要去除无用index了
        # zero_index = []
        # for i in range(self.total_rank):
        #     if all_weight[i][0]==0:
        #         zero_index.append(i)
        # all_index = list(range(self.total_rank))
        # rest_index = [item for item in all_index if item not in zero_index]
        # all_weight = all_weight[rest_index,:]

        # np.save("all_weight.npy",np.array(all_weight.cpu()))
        # correlation = np.zeros((self.total_rank, self.total_rank), dtype=np.float64)  # 奇异值间相关系数
        # slopes = np.zeros((self.total_rank, self.total_rank), dtype=np.float64)  # 奇异值间的斜率
        beta1 = 0.9  # 相关性阈值
        beta2 = 2  # 斜率阈值


        # num_samples = 10
        # num_datasets = 1000
        # X_list = [2 * np.random.rand(num_samples, 1) for _ in range(num_datasets)]
        # tic = time.time()
        # for i in range(num_datasets):
        #     for j in range(num_datasets):
        #         x = X_list[i]
        #         y = X_list[j]
        #         r, _ = st.pearsonr(x.reshape(-1), y.reshape(-1))
        #         if abs(r) > 0.9:
        #             slope, intercept, r_value, p_value, std_err = st.linregress(x.reshape(-1), y.reshape(-1))
        # toc = time.time()
        # time_last = toc - tic
        # print("串行同一个矩阵中两两计算斜率的时间sci：" + str(time_last))
        # end = time.time()



        weight = all_weight.cpu().numpy()
        # np.save("./parallel_test/weight.npy", weight_numpy)

        # 矩阵并行计算皮尔逊相关性系数
        mean_weight = weight.mean(axis=1)
        mean_weight = mean_weight.reshape(len(mean_weight), 1)
        diff = weight - mean_weight
        fenzi = np.matmul(diff, diff.T)  # 皮尔逊相关性系数的所有分子， 是一个1080*1080的矩阵

        diff_pow2 = diff ** 2
        fenmu_x = diff_pow2.sum(axis=1)  # 分母的其中一项
        fenmu_x = fenmu_x.reshape(len(fenmu_x), 1)
        fenmu_corr = np.matmul(fenmu_x, fenmu_x.T)
        fenmu_corr = fenmu_corr ** 0.5 + 1e-15    # 皮尔逊相关性系数的所有分母，是一个1080*1080的矩阵

        correlation = fenzi / fenmu_corr  # 最终的所有皮尔逊相关性系数


        # 并行计算斜率

        # 分子和计算皮尔逊相关性系数的分子是一样的
        fenmu_slope = (diff ** 2).sum(axis=1)
        fenmu_slope = fenmu_slope.reshape(len(fenmu_slope), 1) +  1e-15  # 防止除0错
        slopes = fenzi / fenmu_slope

        # # 将slopes矩阵进行均值池化
        # mean_pooling = []
        # matrix = slopes
        # window_size = 3
        # strip = 3
        # # 遍历矩阵，应用窗口
        # for i in range(0,matrix.shape[0] - window_size + 1, strip):
        #     for j in range(0, matrix.shape[1] - window_size + 1, strip):
        #         window = matrix[i:i + window_size, j:j + window_size]
        #         mean_value = np.mean(window)
        #         mean_pooling.append(mean_value)
        #
        # # 将一维数组转换为矩阵
        # mean_pooling_matrix = np.array(mean_pooling).reshape((matrix.shape[0]//strip, matrix.shape[0]//strip))

        beta1 = 0.9  # 相关性阈值
        beta2 = 2  # 斜率阈值
        mask_cor = correlation < beta1
        mask_slopes = slopes < beta2
        mask = np.logical_or(mask_cor, mask_slopes)
        slopes[mask] = 0



        # 串行计算皮尔逊相关性系数和斜率
        # tic = time.time()
        # for i in rest_index:
        #     for j in rest_index:
        #         weight_i = weight[i]
        #         weight_j = weight[j]
        #         r, _ = st.pearsonr(weight_i, weight_j)
        #         # correlation[i][j] = r
        #         if abs(r) >= beta1:
        #             slope, intercept, r_value, p_value, std_err = st.linregress(weight_i, weight_j)
        #             if slope >= beta2:    # 阈值过滤，未达阈值，斜率设为0
        #                 slopes[i][j] = slope
        # toc = time.time()
        # print(f"串行计算相关性所用时间：{toc - tic}")



        tic = time.time()
        graph = nx.from_numpy_array(slopes,create_using=nx.DiGraph)
        graph = remove_min_edge_in_cycles(graph)
        slopes = nx.to_numpy_array(graph)
        toc = time.time()
        print(f"去环所用时间：{toc - tic}")

        importance = compute_weights(slopes)
        weight = importance / sum(importance)   # 除以总和后，weight的值变得非常小
        weight = weight.reshape(-1,self.lora_init_rank) # 这边不应该是这样，应该是根据奇异值是否为0分配权重。

        index = 0
        for n,p in model.named_parameters():
            if "lora_E" in n:
                ipt = torch.zeros_like(p.data)   # ipt为所有时间窗口内参数权重和
                for i in range(len(self.weight_t)):

                    ipt = ipt + (self.weight_t[i][n]).abs()*regu_weight[i]

                ipt = ipt/sum(ipt)   # ipt的值比weight的值大很多，因此将ipt除以其自身的和
                ipt_score = torch.Tensor(weight[index]).to(ipt.device)+ipt.view(-1)
                # ipt_score = torch.Tensor(weight[index]).to(ipt.device)   # 只考虑因果关联图的权重
                # ipt_score = ipt
                is_dict[n] = ipt_score.view(-1, 1)
                all_is.append(ipt_score.view(-1))
                index = index + 1


        self.weight_t = []
        self.regu_loss = []

        # Calculate the masking threshold
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()  # 第k个最小的元素

        all_name = [name for name in is_dict]
        # Mask out unimportant singular values
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            index = 0 # 打印秩的index
            if Print_rank:
                f = open(f"{root_output_dir}/importance.txt", "a")
                f.write(f"----------------------------step {self.global_step}------------------------------------------\n")
                f.write(f"总秩数为{self.total_rank}, 目标秩数为{self.target_rank}, 当前秩数为{curr_rank}\n")

            for n,p in model.named_parameters():
                if "lora_E" in n:
                    # p.data.masked_fill_(is_dict[]<=mask_threshold, 0.0)  # 将masked_fill_函数变为p*mask矩阵即可。
                    # mask = (is_dict[n]>mask_threshold).to(p.data.dtype)
                    # p.data = p.data * mask  # p.data 乘 mask矩阵
                    ranknum = (is_dict[n]>mask_threshold).sum().item()

                    new_mask = (is_dict[n]>mask_threshold).to(p.data.dtype)
                    # 将orth_losses 损失列表置空

                    self.mask_weight[n] = new_mask
                    # 在这进行mask矩阵的更新。

                    if Print_rank:
                        cur_rank = all_is[index].tolist()
                        f.write("{:70}:".format(all_name[index]))
                        for i in range(self.lora_init_rank):
                            if cur_rank[i] > mask_threshold:
                                f.write(f"{cur_rank[i]}   ")
                            else:
                                f.write(f"[{cur_rank[i]}]   ")
                        f.write("\n")
                    index = index + 1
                    if self.tb_writter is not None and self.global_step%self.log_interval==0:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step)
                        self.rank_pattern[n] = ranknum
                        curr_sum_rank += ranknum
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_A")][1]
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_B")][0]

            if Print_rank:
                f.write(f"阈值为:{mask_threshold}\n")
                f.write(f"排序后的重要性:\n")
                # f.close()
                all_ipts = []
                for ipts in all_is:
                    tmp = ipts.cpu().numpy().tolist()
                    all_ipts.extend(tmp)
                all_ipts.sort(reverse=True)
                for i, ipt in enumerate(all_ipts):
                    if i+1 <= curr_rank:
                        f.write("{:25}".format(ipt))
                    else:
                        f.write("{:25}".format(f"[{ipt}]"))
                    if (i+1) %4 == 0:
                        f.write("\n")
                f.close()

            if self.tb_writter is not None and self.global_step%self.log_interval==0:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return mask_threshold



    def update_and_mask(self, model, global_step,root_output_dir):
        Print_rank = False
        self.global_step = global_step
        if global_step > self.initial_warmup:        # 在秩裁剪完之后，不再计算重要性分数，不裁剪
            # Update importance scores element-wise 
            # self.update_ipt(model, global_step)     #   计算所有参数的敏感度、平滑性分数和不确定性
            self.record_update(model) # 记录参数并更新
            Print_rank = False  #  裁剪结束 ，不打印秩  # 控制是否打印秩。要打印的时候，把这个地方恢复。
            # do not update ipt during final fine-tuning

            # 在裁剪时，才更新mask

        # Budget schedule   计算现有秩数
        curr_rank, mask_ind = self.adaptive_time_window(global_step)   # 计算现有秩数和是否满足了时间窗口数量

        if mask_ind:     # 输入满足了时间窗口数量
            # Mask to target budget 
            mask_threshold = self.mask_to_target_rank(model, curr_rank, Print_rank,root_output_dir)   # 更新mask矩阵
        else:
            mask_threshold = None 
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n,p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov-I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s"%n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                )


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`. 
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param

