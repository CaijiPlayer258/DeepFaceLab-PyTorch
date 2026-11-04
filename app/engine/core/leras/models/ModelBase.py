import types

import numpy as np
import torch
import torch.nn as torch_nn

from core.interact import interact as io
from core.leras import nn

class ModelBase(nn.Saveable, torch_nn.Module):
    """
    PyTorch模型基类
    """
    def __init__(self, *args, name=None, **kwargs):
        torch_nn.Module.__init__(self)
        nn.Saveable.__init__(self, name=name)
        self.layers = []
        self.layers_by_name = {}
        self.built = False
        self.args = args
        self.kwargs = kwargs

    def _build_sub(self, layer, name):
        """递归构建子层"""
        if isinstance(layer, list):
            for i, sublayer in enumerate(layer):
                self._build_sub(sublayer, f"{name}_{i}")
        elif isinstance(layer, dict):
            for subname in layer.keys():
                sublayer = layer[subname]
                self._build_sub(sublayer, f"{name}_{subname}")
        elif isinstance(layer, nn.LayerBase) or isinstance(layer, ModelBase):

            if layer.name is None:
                layer.name = name

            # PyTorch eager：LayerBase 在 __init__ 中已完成权重创建。
            # ModelBase 需要递归 build() 以确保其子层也被发现。
            if isinstance(layer, ModelBase):
                layer.build()

            self.layers.append(layer)
            self.layers_by_name[layer.name] = layer

    def xor_list(self, lst1, lst2):
        """返回两个列表的对称差"""
        return [value for value in lst1+lst2 if (value not in lst1) or (value not in lst2)]

    def build(self):
        """构建模型"""
        # PyTorch不需要variable_scope
        current_vars = []
        generator = None
        while True:

            if generator is None:
                generator = self.on_build(*self.args, **self.kwargs)
                if not isinstance(generator, types.GeneratorType):
                    generator = None

            if generator is not None:
                try:
                    next(generator)
                except StopIteration:
                    generator = None

            # torch.nn.Module 会引入 _modules/_parameters/_buffers 等内部字段。
            # 这些不属于 DFL 的模型层定义，且会导致递归扫描/重复构建。
            v = {
                k: value
                for k, value in vars(self).items()
                if not k.startswith('_')
                and k not in (
                    'layers',
                    'layers_by_name',
                    'built',
                    'args',
                    'kwargs',
                )
            }

            # torch.nn.Module 将子模块存储在 _modules 中（不一定出现在 vars(self)）。
            for k, subm in getattr(self, '_modules', {}).items():
                if k is None:
                    continue
                if k.startswith('_'):
                    continue
                if k in v:
                    continue
                v[k] = subm
            new_vars = self.xor_list(current_vars, list(v.keys()))

            for name in new_vars:
                self._build_sub(v[name], name)

            current_vars += new_vars

            if generator is None:
                break

        self.built = True

    #override
    def get_weights(self):
        if not self.built:
            self.build()

        weights = []
        for layer in self.layers:
            weights += layer.get_weights()
        return weights

    def _get_tf_weight_names(self):
        """Return list of (tf_variable_name, param_tensor) pairs for loading
        original DeepFaceLab (TensorFlow) .npy weights."""
        if not self.built:
            self.build()
        result: list = []
        self._collect_tf_weights("", result)
        return result

    def _collect_tf_weights(self, prefix, result):
        """Recursively collect (tf_name, tensor) pairs from the layer tree.

        Collects ALL parameter attributes (not just weight/bias/running_mean/running_var)
        so eps (FRNorm2D), tau (TLU) etc. are also included for proper TF→Torch conversion.
        """
        for layer in self.layers:
            layer_name = layer.name
            if layer_name is None:
                continue
            layer_prefix = f"{prefix}/{layer_name}" if prefix else layer_name

            if isinstance(layer, nn.LayerBase):
                for attr in ("weight", "bias", "running_mean", "running_var", "eps", "tau"):
                    t = getattr(layer, attr, None)
                    if t is not None and (isinstance(t, torch.nn.Parameter) or isinstance(t, torch.Tensor)):
                        tf_name = f"{layer_prefix}/{attr}:0"
                        result.append((tf_name, t))
            elif isinstance(layer, ModelBase):
                layer._collect_tf_weights(layer_prefix, result)

    def get_layer_by_name(self, name):
        return self.layers_by_name.get(name, None)

    def get_layers(self):
        """获取所有层"""
        if not self.built:
            self.build()
        layers = []
        for layer in self.layers:
            if isinstance(layer, nn.LayerBase):
                layers.append(layer)
            else:
                layers += layer.get_layers()
        return layers

    def on_build(self, *args, **kwargs):
        """
        在这里初始化模型层
        
        如果构建未完成，返回'yield'，这样依赖的模型会被初始化
        """
        pass

    def forward(self, *args, **kwargs):
        """前向传播：在这里组织层/模型/张量的流动"""
        pass

    def __call__(self, *args, **kwargs):
        """调用模型"""
        if not self.built:
            self.build()

        return self.forward(*args, **kwargs)

    # def compute_output_shape(self, shapes):
    #     if not self.built:
    #         self.build()

    #     not_list = False
    #     if not isinstance(shapes, list):
    #         not_list = True
    #         shapes = [shapes]

    #     with tf.device('/CPU:0'):
    #         # CPU tensors will not impact any performance, only slightly RAM "leakage"
    #         phs = []
    #         for dtype,sh in shapes:
    #             phs += [ tf.placeholder(dtype, sh) ]

    #         result = self.__call__(phs[0] if not_list else phs)

    #         if not isinstance(result, list):
    #             result = [result]

    #         result_shapes = []

    #         for t in result:
    #             result_shapes += [ t.shape.as_list() ]

    #         return result_shapes[0] if not_list else result_shapes

    def build_for_run(self, shapes_list):
        """
        PyTorch版本：不需要placeholder，直接准备运行
        shapes_list只用于验证
        """
        if not isinstance(shapes_list, list):
            raise ValueError("shapes_list必须是列表")
        
        # PyTorch是eager模式，不需要预先构建计算图
        # 只是确保模型已构建
        if not self.built:
            self.build()

    def run(self, inputs):
        """
        PyTorch版本：直接前向传播
        """
        if not self.built:
            self.build()
        
        # PyTorch直接调用forward
        import torch
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # 确保输入是tensor
        tensor_inputs = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                tensor_inputs.append(torch.from_numpy(inp))
            else:
                tensor_inputs.append(inp)
        
        return self.__call__(*tensor_inputs)

    def summary(self):
        """打印模型摘要"""
        layers = self.get_layers()
        layers_names = []
        layers_params = []

        max_len_str = 0
        max_len_param_str = 0
        delim_str = "-"

        total_params = 0

        # 获取层名称和字符串长度用于分隔符
        for l in layers:
            if len(str(l)) > max_len_str:
                max_len_str = len(str(l))
            layers_names += [str(l).capitalize()]

        # 获取每层的参数数量
        layers_params = [int(np.sum(np.prod(w.shape) for w in l.get_weights())) for l in layers]
        total_params = np.sum(layers_params)

        # 获取字符串长度用于分隔符
        for p in layers_params:
            if len(str(p)) > max_len_param_str:
                max_len_param_str = len(str(p))

        # 设置分隔符
        for i in range(max_len_str+max_len_param_str+3):
            delim_str += "-"

        output = "\n"+delim_str+"\n"

        # 格式化模型名称字符串
        model_name_str = "| "+self.name.capitalize()
        len_model_name_str = len(model_name_str)
        for i in range(len(delim_str)-len_model_name_str):
            model_name_str += " " if i != (len(delim_str)-len_model_name_str-2) else " |"

        output += model_name_str +"\n"
        output += delim_str +"\n"

        # 格式化层表格
        for i in range(len(layers_names)):
            output += delim_str +"\n"

            l_name = layers_names[i]
            l_param = str(layers_params[i])
            l_param_str = ""
            if len(l_name) <= max_len_str:
                for i in range(max_len_str - len(l_name)):
                    l_name += " "

            if len(l_param) <= max_len_param_str:
                for i in range(max_len_param_str - len(l_param)):
                    l_param_str += " "

            l_param_str += l_param

            output += "| "+l_name+"|"+l_param_str+"| \n"

        output += delim_str +"\n"

        # 格式化参数总和
        total_params_str = "| 总参数数量: "+str(total_params)
        len_total_params_str = len(total_params_str)
        for i in range(len(delim_str)-len_total_params_str):
            total_params_str += " " if i != (len(delim_str)-len_total_params_str-2) else " |"

        output += total_params_str +"\n"
        output += delim_str +"\n"

        io.log_info(output)

nn.ModelBase = ModelBase
