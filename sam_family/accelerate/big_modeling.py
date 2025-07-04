# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .hooks import (
    AlignDevicesHook,
    CpuOffload,
    UserCpuOffloadHook,
    add_hook_to_module,
    attach_align_device_hook,
    attach_align_device_hook_on_blocks,
)
from .utils import (
    OffloadedWeightsLoader,
    check_cuda_p2p_ib_support,
    check_device_map,
    extract_submodules_state_dict,
    find_tied_parameters,
    get_balanced_memory,
    infer_auto_device_map,
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_torch_version,
    is_xpu_available,
    load_checkpoint_in_model,
    load_checkpoint_in_model_hyperload,
    offload_state_dict,
    parse_flag_from_env,
    retie_parameters,
)
from .utils.other import recursive_getattr


logger = logging.getLogger(__name__)


@contextmanager
def init_empty_weights(include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

    # TODO(shingjan): remove the torch version check once older versions are deprecated
    if is_torch_version(">=", "2.0") and include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def cpu_offload(
    model: nn.Module,
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    preload_module_classes: Optional[List[str]] = None,
):
    """
    Activates full CPU offload for a model. As a result, all parameters of the model will be offloaded and only one
    copy of the state dict of the model will be kept. During the forward pass, parameters will be extracted from that
    state dict and put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the model that will be kept on CPU.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    if state_dict is None:
        state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}

    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=state_dict,
        preload_module_classes=preload_module_classes,
    )

    return model


def cpu_offload_with_hook(
    model: torch.nn.Module,
    execution_device: Optional[Union[int, str, torch.device]] = None,
    prev_module_hook: Optional[UserCpuOffloadHook] = None,
):
    """
    Offloads a model on the CPU and puts it back to an execution device when executed. The difference with
    [`cpu_offload`] is that the model stays on the execution device after the forward and is only offloaded again when
    the `offload` method of the returned `hook` is called. Useful for pipelines running a model in a loop.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by this function for a previous model in the pipeline you are running. If passed, its
            offload method will be called just before the forward of the model to which this hook is attached.

    Example:

    ```py
    model_1, hook_1 = cpu_offload_with_hook(model_1, cuda_device)
    model_2, hook_2 = cpu_offload_with_hook(model_2, cuda_device, prev_module_hook=hook_1)
    model_3, hook_3 = cpu_offload_with_hook(model_3, cuda_device, prev_module_hook=hook_2)

    hid_1 = model_1(input)
    for i in range(50):
        # model1 is offloaded on the CPU at the first iteration, model 2 stays on the GPU for this whole loop.
        hid_2 = model_2(hid_1)
    # model2 is offloaded to the CPU just before this forward.
    hid_3 = model_3(hid_3)

    # For model3, you need to manually call the hook offload method.
    hook_3.offload()
    ```
    """
    hook = CpuOffload(execution_device=execution_device, prev_module_hook=prev_module_hook)
    add_hook_to_module(model, hook, append=True)
    user_hook = UserCpuOffloadHook(model, hook)
    return model, user_hook


def disk_offload(
    model: nn.Module,
    offload_dir: Union[str, os.PathLike],
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    preload_module_classes: Optional[List[str]] = None,
):
    """
    Activates full disk offload for a model. As a result, all parameters of the model will be offloaded as
    memory-mapped array in a given folder. During the forward pass, parameters will be accessed from that folder and
    put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`): The model to offload.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model's first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")):
        offload_state_dict(offload_dir, model.state_dict())
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    weights_map = OffloadedWeightsLoader(save_folder=offload_dir)

    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        model,
        execution_device=execution_device,
        offload=True,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
        preload_module_classes=preload_module_classes,
    )

    return model

def extract_submodules_state_dict_B(
        state_dict: Dict[str, torch.Tensor],
        submodule_names: List[str],
        LayerName2Index : Dict
):
    """
    Extract the sub state-dict corresponding to a list of given submodules.

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dict to extract from.
        submodule_names (`List[str]`): The list of submodule names we want to extract.
    """

    result_index = {}
    for module_name in submodule_names:
        result_index.update(
            {
                key: index_params
                for key, index_params in LayerName2Index.items()
                if key == module_name or key.startswith(module_name + ".")
            }
        )

    return result_index


def extract_submodules_state_dict_C(
        state_dict: Dict[str, torch.Tensor],
        submodule_names: List[str],
        LayerName2Index : Dict
):
    """
    Extract the sub state-dict corresponding to a list of given submodules.

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dict to extract from.
        submodule_names (`List[str]`): The list of submodule names we want to extract.
    """
    # result = {}
    # for module_name in submodule_names:
    #     # We want to catch module_name parameter (module_name.xxx) or potentially module_name, but not any of the
    #     # submodules that could being like module_name (transformers.h.1 and transformers.h.10 for instance)
    #     result.update(
    #         {
    #             key: param
    #             for key, param in state_dict.items()
    #             if key == module_name or key.startswith(module_name + ".")
    #         }
    #     )

    result_index = {}
    for module_name in submodule_names:
        # We want to catch module_name parameter (module_name.xxx) or potentially module_name, but not any of the
        # submodules that could being like module_name (transformers.h.1 and transformers.h.10 for instance)
        result_index.update(
            {
                key: index_params
                for key, index_params in LayerName2Index.items()
                if key == module_name or key.startswith(module_name + ".")
            }
        )

    return result_index


def dispatch_model(
    model: nn.Module,
    device_map: Dict[str, Union[str, int, torch.device]],
    main_device: Optional[torch.device] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    offload_dir: Optional[Union[str, os.PathLike]] = None,
    offload_index: Optional[Dict[str, str]] = None,
    offload_buffers: bool = False,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    force_hooks: bool = False,
    hyper_compress: bool = False,
    hyper_inference_mode: Optional[str] = None,
    hyper_save_path: Optional[str] = None,
    model_name : Optional[str] = None,
    LayerName2Index : Optional[Dict] = None,
    rect_l: Optional = None,
    num_inner_list: Optional = None
):
    """
    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype`/ `shape` or safetensors filename). Will default
            to the index saved in `save_folder`.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
    """
    # Error early if the device map is incomplete.
    check_device_map(model, device_map)

    # for backward compatibility
    is_bnb_quantized = (
        getattr(model, "is_quantized", False) or getattr(model, "is_loaded_in_8bit", False)
    ) and getattr(model, "quantization_method", "bitsandbytes") == "bitsandbytes"

    # We attach hooks if the device_map has at least 2 different devices or if
    # force_hooks is set to `True`. Otherwise, the model in already loaded
    # in the unique device and the user can decide where to dispatch the model.
    # If the model is quantized, we always force-dispatch the model


    if (len(set(device_map.values())) > 1) or is_bnb_quantized or force_hooks or hyper_compress:
        if main_device is None:
            if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
                main_device = "cpu"
            else:
                main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]


        if hyper_compress and hyper_inference_mode == 'C':
            main_device = 0
        elif hyper_compress and hyper_inference_mode in ['A', 'B']:
            main_device = 'cpu'


        if main_device != "cpu" or hyper_inference_mode in ['A', 'B', 'C']:
            cpu_modules = [name for name, device in device_map.items() if device == "cpu"]

            if state_dict is None and len(cpu_modules) > 0:
                if hyper_compress and hyper_inference_mode == 'B':
                    state_dict = extract_submodules_state_dict_B(model.state_dict(), cpu_modules, LayerName2Index)
                elif hyper_compress and hyper_inference_mode == 'C':
                    state_dict = extract_submodules_state_dict_C(model.state_dict(), cpu_modules, LayerName2Index)
                elif hyper_compress and hyper_inference_mode == 'A':
                    state_dict = extract_submodules_state_dict_C(model.state_dict(), cpu_modules, LayerName2Index)
                elif not hyper_compress:
                    state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)


        disk_modules = [name for name, device in device_map.items() if device == "disk"]
        if offload_dir is None and offload_index is None and len(disk_modules) > 0:
            raise ValueError(
                "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
                f"need to be offloaded: {', '.join(disk_modules)}."
            )
        if (
            len(disk_modules) > 0
            and offload_index is None
            and (not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")))
        ):
            disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
            offload_state_dict(offload_dir, disk_state_dict)

        execution_device = {
            name: main_device if device in ["cpu", "disk"] else device for name, device in device_map.items()
        }
        execution_device[""] = main_device
        offloaded_devices = ["disk"] if main_device == "cpu" or main_device == "mps" else ["cpu", "disk"]

        if hyper_inference_mode in ['A', 'B']:
            offload = {name: True for name, device in device_map.items()}
        else:
            offload = {name: device in offloaded_devices for name, device in device_map.items()}

        save_folder = offload_dir if len(disk_modules) > 0 else None
        if state_dict is not None or save_folder is not None or offload_index is not None:
            device = main_device if offload_index is not None else None
            weights_map = OffloadedWeightsLoader(
                state_dict=state_dict, save_folder=save_folder, index=offload_index, device=device
            )
        else:
            weights_map = None

        # When dispatching the model's parameters to the devices specified in device_map, we want to avoid allocating memory several times for the
        # tied parameters. The dictionary tied_params_map keeps track of the already allocated data for a given tied parameter (represented by its
        # original pointer) on each devices.
        tied_params = find_tied_parameters(model)

        tied_params_map = {}
        for group in tied_params:
            for param_name in group:
                # data_ptr() is enough here, as `find_tied_parameters` finds tied params simply by comparing `param1 is param2`, so we don't need
                # to care about views of tensors through storage_offset.
                data_ptr = recursive_getattr(model, param_name).data_ptr()
                tied_params_map[data_ptr] = {}

                # Note: To handle the disk offloading case, we can not simply use weights_map[param_name].data_ptr() as the reference pointer,
                # as we have no guarantee that safetensors' `file.get_tensor()` will always give the same pointer.


        if hyper_compress:
            attach_align_device_hook_on_blocks(
                model,
                execution_device=execution_device,
                offload=offload,
                offload_buffers=offload_buffers,
                weights_map=weights_map,
                skip_keys=skip_keys,
                preload_module_classes=preload_module_classes,
                tied_params_map=tied_params_map,
                hyper_save_path=hyper_save_path,
                model_name=model_name,
                hyper_inference_mode = hyper_inference_mode,
                rect_l = rect_l,
                num_inner_list = num_inner_list
            )
        else:
            attach_align_device_hook_on_blocks(
                model,
                execution_device=execution_device,
                offload=offload,
                offload_buffers=offload_buffers,
                weights_map=weights_map,
                skip_keys=skip_keys,
                preload_module_classes=preload_module_classes,
                tied_params_map=tied_params_map,
                model_name=model_name
            )

        # warn if there is any params on the meta device
        offloaded_devices_str = " and ".join(
            [device for device in set(device_map.values()) if device in ("cpu", "disk")]
        )
        # if len(offloaded_devices_str) > 0:
        #     logger.warning(
        #         f"Some parameters are on the meta device because they were offloaded to the {offloaded_devices_str}."
        #     )

        # Attaching the hook may break tied weights, so we retie them
        retie_parameters(model, tied_params)

        # add warning to cuda and to method
        def add_warning(fn, model):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                warning_msg = "You shouldn't move a model that is dispatched using accelerate hooks."
                if str(fn.__name__) == "to":
                    to_device = torch._C._nn._parse_to(*args, **kwargs)[0]
                    if to_device is not None:
                        logger.warning(warning_msg)
                else:
                    logger.warning(warning_msg)
                for param in model.parameters():
                    if param.device == torch.device("meta"):
                        raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
                return fn(*args, **kwargs)

            return wrapper

        # Make sure to update _accelerate_added_attributes in hooks.py if you add any hook
        model.to = add_warning(model.to, model)
        if is_npu_available():
            model.npu = add_warning(model.npu, model)
        elif is_mlu_available():
            model.mlu = add_warning(model.mlu, model)
        elif is_musa_available():
            model.musa = add_warning(model.musa, model)
        elif is_xpu_available():
            model.xpu = add_warning(model.xpu, model)
        else:
            model.cuda = add_warning(model.cuda, model)

        # Check if we are using multi-gpus with RTX 4000 series
        use_multi_gpu = len([device for device in set(device_map.values()) if device not in ("cpu", "disk")]) > 1
        if use_multi_gpu and not check_cuda_p2p_ib_support():
            logger.warning(
                "We've detected an older driver with an RTX 4000 series GPU. These drivers have issues with P2P. "
                "This can affect the multi-gpu inference when using accelerate device_map."
                "Please make sure to update your driver to the latest version which resolves this."
            )
    else:
        device = list(device_map.values())[0] # 若device只有1个, 则model就整个放在那个device上面，不dispatch
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if is_npu_available() and isinstance(device, int):
            device = f"npu:{device}"
        elif is_mlu_available() and isinstance(device, int):
            device = f"mlu:{device}"
        elif is_musa_available() and isinstance(device, int):
            device = f"musa:{device}"
        elif is_xpu_available() and isinstance(device, int):
            device = f"xpu:{device}"

        if device != "disk":
            model.to(device)
        else:
            raise ValueError(
                "You are trying to offload the whole model to the disk. Please use the `disk_offload` function instead."
            )
    # Convert OrderedDict back to dict for easier usage
    model.hf_device_map = dict(device_map)
    # LayerName2Index
    return model


def load_checkpoint_and_dispatch(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = None,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    offload_buffers: bool = False,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: Optional[bool] = None,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    force_hooks: bool = False,
    strict: bool = False,
    hyper_compress: bool = False,
    hyper_inference_mode: Optional[str] = None,
    model_name: Optional[str] = None
):

    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model (`torch.nn.Module`): The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For more
            information about each option see [here](../concept_guides/big_model_inference#designing-a-device-map).
            Defaults to None, which means [`dispatch_model`] will not be called.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the checkpoint state_dict match the keys of the model's
            state_dict.

    """
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError(
            "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
            "'sequential'."
        )
    if isinstance(device_map, str):
        if device_map != "sequential":

            if not hyper_compress:
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    dtype=dtype,
                    low_zero=(device_map == "balanced_low_0"),
                )


        if hyper_compress and hyper_inference_mode in ['A', 'B', 'C']:
            device_map = collections.OrderedDict()
            for name, _ in model.named_children():
                device_map[name] = 'cpu'

    if offload_state_dict is None and device_map is not None and "disk" in device_map.values():
        offload_state_dict = True



    ############### edit #####################
    if hyper_compress:
        LayerName2Index, rect_l, num_inner_list = load_checkpoint_in_model(
            model,
            checkpoint,
            device_map=device_map,
            offload_folder=offload_folder,
            dtype=dtype,
            offload_state_dict=offload_state_dict,
            offload_buffers=offload_buffers,
            strict=strict,
            hyper_compress=hyper_compress,
            hyper_inference_mode=hyper_inference_mode,
            model_name=model_name
        )
    else:
        LayerName2Index, rect_l, num_inner_list = load_checkpoint_in_model(
            model,
            checkpoint,
            device_map=device_map,
            offload_folder=offload_folder,
            dtype=dtype,
            offload_state_dict=offload_state_dict,
            offload_buffers=offload_buffers,
            strict=strict,
            hyper_compress = hyper_compress,
            hyper_inference_mode = hyper_inference_mode
        )

    compressed_tensor_name = list(LayerName2Index.keys())
    empty_tensor_name = list(model.state_dict())
    error = 0
    for i in empty_tensor_name:
        if i not in compressed_tensor_name:
            error += 1

    for j in list(model.named_buffers()):
        name_buffer = j[0]
        tensor_buffer = j[1]
        if name_buffer not in empty_tensor_name:
            print(f"{name_buffer} : {tensor_buffer}")


    if device_map is None:
        return model

    if hyper_compress:
        return dispatch_model(
            model,
            device_map=device_map,
            offload_dir=offload_folder,
            offload_buffers=offload_buffers,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            force_hooks=force_hooks,
            hyper_compress=hyper_compress,
            hyper_inference_mode=hyper_inference_mode,
            hyper_save_path = checkpoint,
            model_name=model_name,
            LayerName2Index = LayerName2Index,
            rect_l = rect_l,
            num_inner_list = num_inner_list
        )
    else:
        return dispatch_model(
            model,
            device_map=device_map,
            offload_dir=offload_folder,
            offload_buffers=offload_buffers,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            force_hooks=force_hooks,
            hyper_compress=hyper_compress,
            hyper_inference_mode=hyper_inference_mode,
            model_name=model_name
        )



def load_checkpoint_and_dispatch_hyperload(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = None,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    offload_buffers: bool = False,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: Optional[bool] = None,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    force_hooks: bool = False,
    strict: bool = False,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model (`torch.nn.Module`): The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For more
            information about each option see [here](../concept_guides/big_model_inference#designing-a-device-map).
            Defaults to None, which means [`dispatch_model`] will not be called.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the checkpoint state_dict match the keys of the model's
            state_dict.
    """

    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError(
            "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
            "'sequential'."
        )
    if isinstance(device_map, str):
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
            dtype=dtype,
            offload_buffers=offload_buffers,
        )
    if offload_state_dict is None and device_map is not None and "disk" in device_map.values():
        offload_state_dict = True


    load_checkpoint_in_model_hyperload(
        model,
        checkpoint,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype=dtype,
        offload_state_dict=offload_state_dict,
        offload_buffers=offload_buffers,
        strict=strict,
    )
    if device_map is None:
        return model
    return dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
        offload_buffers=offload_buffers,
        skip_keys=skip_keys,
        preload_module_classes=preload_module_classes,
        force_hooks=force_hooks,
    )