from typing import Optional
import torch
import platform
osplatform = platform.system()
import numpy as np
from .accelerate import init_empty_weights
from .accelerate import load_checkpoint_and_dispatch



def load_model_dipatch(checkpoint:str, model_name:str, rect_l:float, hyper_inference_mode:str,
                       sam_model_registry, model_type):
    # decoding files path
    # Decode_Param_Path = checkpoint + f'/{model_name}_l_{str(rect_l)[0] + str(rect_l)[2:]}'

    checkpoint_set = checkpoint + '/Compressed_Dir'
    max_memory_set = None
    hyper_compress_set = True

    with init_empty_weights():
        model = sam_model_registry[model_type]()


    sam = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_set,
        device_map="auto",
        max_memory=max_memory_set,
        offload_folder='offload',
        offload_state_dict=True,
        no_split_module_classes=['Block'],
        offload_buffers=True,
        hyper_compress=hyper_compress_set,
        hyper_inference_mode=hyper_inference_mode,
        model_name=model_name
    )
    return sam

class SegAny:
    def __init__(self, model_name:str, checkpoint:str, hyper_load: bool, use_bfloat16:bool=False, device:str='None', hyper_file:Optional=None, eval_quant:Optional=False):
        print('--' * 20)
        print('* Init SAM... *')
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.model_source = None
        self.device = device

        if model_name == 'mobile_sam':
            # mobile sam
            from sam_family.mobile_sam import sam_model_registry, SamPredictor
            self.model_type = "vit_t"
            self.model_source = 'mobile_sam'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)


        elif model_name == 'hyper_mobile_sam':
            from sam_family.mobile_sam import sam_model_registry, SamPredictor
            self.model_type = "vit_t"
            self.model_source = 'mobile_sam'
            rect_l = 0.5
            sam = load_model_dipatch(checkpoint=checkpoint, model_name=model_name, rect_l=rect_l,
                                     hyper_inference_mode='A', sam_model_registry=sam_model_registry,
                                     model_type=self.model_type)
            self.model = sam
            self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'efficient_sam_ti':
            from sam_family.efficient_sam.build_efficient_sam import build_efficient_sam_vitt

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = build_efficient_sam_vitt(checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = self.model
            else:
                self.model = build_efficient_sam_vitt(checkpoint)
                self.model.to(device=self.device)
                self.predictor_with_point_prompt = build_efficient_sam_vitt(checkpoint)

        elif model_name == 'efficient_sam_s':
            from sam_family.efficient_sam.build_efficient_sam import build_efficient_sam_vits

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = build_efficient_sam_vits(checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = self.model
            else:
                self.model = build_efficient_sam_vits(checkpoint)
                self.model.to(device=self.device)
                self.predictor_with_point_prompt = build_efficient_sam_vits(checkpoint)

        elif model_name == 'tiny_sam':
            import sys
            sys.path.append("..")
            from sam_family.tinysam import sam_model_registry, SamPredictor
            self.model_type = "vit_t"

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'med_sam':
            from sam_family.MedSAM.segment_anything import sam_model_registry
            medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
            medsam_model = medsam_model.to(self.device)
            medsam_model.eval()
            self.model = medsam_model
            self.predictor_with_point_prompt = medsam_model



        elif model_name == 'mobile_sam_v2_h':
            from sam_family.MobileSAMv2.mobilesamv2 import sam_model_registry, SamPredictor

            encoder_type = 'sam_vit_h'

            try:
                Prompt_guided_path = 'sam_family/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'

                PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
                mobilesamv2 = sam_model_registry['vit_h']()
                mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
                mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']

                image_encoder = sam_model_registry[encoder_type](checkpoint)
                mobilesamv2.image_encoder = image_encoder
            except:
                PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder']()
                mobilesamv2 = sam_model_registry['vit_h']()
                mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
                mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']

                image_encoder = sam_model_registry[encoder_type]()
                mobilesamv2.image_encoder = image_encoder
                with open(checkpoint, "rb") as f:
                    state_dict = torch.load(f)
                mobilesamv2.load_state_dict(state_dict, strict=True)

            mobilesamv2.to(device=self.device)
            mobilesamv2.eval()

            self.model = mobilesamv2
            self.predictor_with_point_prompt = SamPredictor(mobilesamv2)

        elif model_name == 'mobile_sam_v2_l2':
            from sam_family.MobileSAMv2.mobilesamv2 import sam_model_registry, SamPredictor

            encoder_type = 'efficientvit_l2'

            try:
                Prompt_guided_path = 'sam_family/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
                PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
                mobilesamv2 = sam_model_registry['vit_h']()
                mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
                mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']

                image_encoder = sam_model_registry[encoder_type](checkpoint)
                mobilesamv2.image_encoder = image_encoder

            except:
                PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder']()
                mobilesamv2 = sam_model_registry['vit_h']()
                mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
                mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']

                image_encoder = sam_model_registry[encoder_type]()
                mobilesamv2.image_encoder = image_encoder
                with open(checkpoint, "rb") as f:
                    state_dict = torch.load(f)
                mobilesamv2.load_state_dict(state_dict, strict=True)


            mobilesamv2.to(device=self.device)
            mobilesamv2.eval()
            self.model = mobilesamv2
            self.predictor_with_point_prompt = SamPredictor(mobilesamv2)


        elif model_name == 'sam_hq_vit_b':
            # sam hq
            from sam_family.segment_anything_hq import sam_model_registry, SamPredictor
            self.model_type = "vit_b"
            self.model_source = 'sam_hq'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)



        elif model_name == 'hyper_sam_hq_vit_b':
            # sam hq
            from sam_family.segment_anything_hq import sam_model_registry, SamPredictor
            self.model_type = "vit_b"
            self.model_source = 'sam_hq'
            rect_l = 0.1
            sam = load_model_dipatch(checkpoint=checkpoint, model_name=model_name, rect_l=rect_l,
                                     hyper_inference_mode='A', sam_model_registry=sam_model_registry,
                                     model_type=self.model_type)
            self.model = sam
            self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam_hq_vit_h':
            # sam hq
            from sam_family.segment_anything_hq import sam_model_registry, SamPredictor
            self.model_type = "vit_h"
            self.model_source = 'sam_hq'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam_hq_vit_l':
            # sam hq
            from sam_family.segment_anything_hq import sam_model_registry, SamPredictor
            self.model_type = "vit_l"
            self.model_source = 'sam_hq'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam_hq_vit_tiny':
            # sam hq
            from sam_family.segment_anything_hq import sam_model_registry, SamPredictor
            self.model_type = "vit_tiny"
            self.model_source = 'sam_hq'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'edge_sam':
            # edge_sam
            from sam_family.edge_sam import sam_model_registry, SamPredictor
            self.model_type = "edge_sam"
            self.model_source = 'edge_sam'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'edge_sam_3x':
            # edge_sam
            from sam_family.edge_sam import sam_model_registry, SamPredictor
            self.model_type = "edge_sam"
            self.model_source = 'edge_sam'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam_vit_b':
            # sam
            if torch.__version__ > '2.1.1' and osplatform == 'Linux':
                from sam_family.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            else:
                from sam_family.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            self.model_type = "vit_b"
            self.model_source = 'sam'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam_vit_h':
            # sam
            if torch.__version__ > '2.1.1' and osplatform == 'Linux':
                from sam_family.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            else:
                from sam_family.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            self.model_type = "vit_h"
            self.model_source = 'sam'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam_vit_l':
            # sam
            if torch.__version__ > '2.1.1' and osplatform == 'Linux':
                from sam_family.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            else:
                from sam_family.segment_anything import sam_model_registry, SamPredictor
                print('segment_anything')
            self.model_type = "vit_l"
            self.model_source = 'sam'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam2_tiny':
            from sam_family.sam2.build_sam import sam_model_registry
            from sam_family.sam2.sam2_image_predictor import SAM2ImagePredictor as SamPredictor
            self.model_type = "sam2_hiera_tiny"
            self.model_source = 'sam2'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam2_small':
            from sam_family.sam2.build_sam import sam_model_registry
            from sam_family.sam2.sam2_image_predictor import SAM2ImagePredictor as SamPredictor
            self.model_type = "sam2_hiera_small"
            self.model_source = 'sam2'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam2_base_plus':
            from sam_family.sam2.build_sam import sam_model_registry
            from sam_family.sam2.sam2_image_predictor import SAM2ImagePredictor as SamPredictor
            self.model_type = 'sam2_hiera_base_plus'
            self.model_source = 'sam2'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)

        elif model_name == 'sam2_large':
            from sam_family.sam2.build_sam import sam_model_registry
            from sam_family.sam2.sam2_image_predictor import SAM2ImagePredictor as SamPredictor
            self.model_type = "sam2_hiera_large"
            self.model_source = 'sam2'

            if hyper_load:
                from utils.model_hyperload import load_from_hyperencoded
                from utils.model_decode import decompress_params
                with init_empty_weights():
                    sam = sam_model_registry[self.model_type](checkpoint=None)
                decode_mode = 'hyper_op'
                model_layer_names = list(sam.state_dict().keys())
                hyper_dict = decompress_params(model_layer_names, hyper_file, decode_mode, self.device)
                hyper_sam = load_from_hyperencoded(sam, hyper_dict, device)
                hyper_sam = hyper_sam.eval()
                self.model = hyper_sam
                self.predictor_with_point_prompt = SamPredictor(hyper_sam)
            elif eval_quant:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor
            else:
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.eval().to(self.model_dtype)
                sam.to(device=self.device)
                self.model = sam
                self.predictor_with_point_prompt = SamPredictor(sam)


        torch.cuda.reset_peak_memory_stats()

        torch.cuda.empty_cache()

        print('  - device  : {}'.format(self.device))
        print('  - dtype   : {}'.format(self.model_dtype))
        print('  - loading : {}'.format(checkpoint))
        print('  - hyper_load : {}'.format(hyper_load))

        print('* Init Model finished *')
        print('--'*20)
        self.image = None


    def set_image(self, image):
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype):
            self.image = image
            self.predictor_with_point_prompt.set_image(image)

    def reset_image(self):
        self.predictor_with_point_prompt.reset_image()
        self.image = None
        torch.cuda.empty_cache()

    def binary_iou(self, s, g):
        # 两者相乘值为1的部分为交集
        intersecion = np.multiply(s, g)
        # 两者相加，值大于0的部分为交集
        union = np.asarray(s + g > 0, np.float32)
        iou = intersecion.sum() / (union.sum() + 1e-10)
        return iou


    def predict_with_point_prompt(self, input_point, input_label):
        import time
        ##### edit #####
        time_start = time.perf_counter()

        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype):

            if 'sam2' not in self.model_type:
                input_point = np.array(input_point)
                input_label = np.array(input_label)
            else:
                input_point = input_point
                input_label = input_label

            masks, scores, logits = self.predictor_with_point_prompt.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            if self.model_source == 'sam_med2d':
                return masks

            mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            masks, _, _ = self.predictor_with_point_prompt.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )

            return masks


    def predict_with_box_prompt(self, box):
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.model_dtype):
            masks, scores, logits = self.predictor_with_point_prompt.predict(
                box=box,
                multimask_output=False,
            )
            torch.cuda.empty_cache()
            return masks