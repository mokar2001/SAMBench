"""SAM 3 PVS adapter: cached image features, single boxes, and point-grid AMG.

The official Hugging Face Sam3TrackerModel is the visual instance-segmentation
head of facebook/sam3. No detector, text prompt, or ground-truth mask is used.
The pinned SAM 2 AMG supplies the same grid/crops/filtering/NMS as SAM 2/2.1;
every mask and quality estimate comes from SAM 3.
"""
from importlib.metadata import version
import numpy as np


class Sam3Predictor:
    def __init__(self, checkpoint, device, expected_version='5.17.0'):
        import torch
        try:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as exc:
            raise ValueError('Install SAM 3 dependencies: .venv/bin/python -m pip install -r requirements-sam3.txt') from exc
        if version('transformers')!=expected_version:
            raise ValueError(f'SAM 3 requires transformers=={expected_version}; install requirements-sam3.txt.')
        model, loading = Sam3TrackerModel.from_pretrained(
            str(checkpoint.parent), local_files_only=True, dtype=torch.float32,
            attn_implementation='sdpa', output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError(f'Incomplete SAM 3 tracker weights: {loading}')
        self.model=model.to(device).eval()
        self.model.mask_decoder.dynamic_multimask_via_stability=False
        self.model.config.mask_decoder_config.dynamic_multimask_via_stability=False
        self.processor=Sam3TrackerProcessor.from_pretrained(str(checkpoint.parent),local_files_only=True)
        # Prompt coordinates live in this model's native resized image frame.
        self.image_size=self.processor.target_size
        from sam2.utils.transforms import SAM2Transforms
        self._transforms=SAM2Transforms(self.image_size,mask_threshold=0.0,max_hole_area=0.0,max_sprinkle_area=0.0)
        self.mask_threshold=0.0
        self.reset_predictor()

    @property
    def device(self):
        return self.model.device

    def reset_predictor(self):
        self._features=None
        self._orig_hw=None

    def set_image(self, image):
        import torch
        self.reset_predictor()
        if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Expected an HWC RGB image.')
        inputs=self.processor(images=image,return_tensors='pt')
        with torch.inference_mode():
            self._features=self.model.get_image_embeddings(inputs['pixel_values'].to(self.device,dtype=torch.float32))
        self._orig_hw=tuple(image.shape[:2])

    def _predict(self, point_coords=None, point_labels=None, boxes=None, mask_input=None,
                 multimask_output=True, return_logits=False, **kwargs):
        """AMG interface: prompts are already in the resized model coordinate frame."""
        import torch
        if self._features is None:
            raise ValueError('Call set_image before predicting.')
        if mask_input is not None:
            raise ValueError('Mask-prompt refinement is outside this benchmark protocol.')
        inputs={'image_embeddings':self._features,'multimask_output':multimask_output}
        if point_coords is not None:
            inputs['input_points']=point_coords.unsqueeze(0)
            inputs['input_labels']=point_labels.unsqueeze(0)
        if boxes is not None:
            inputs['input_boxes']=boxes.reshape(1,-1,4)
        with torch.inference_mode():
            output=self.model(**inputs)
            masks=self.processor.post_process_masks(
                output.pred_masks, [self._orig_hw], mask_threshold=0.0,
                binarize=False, max_hole_area=0.0,max_sprinkle_area=0.0,
                apply_non_overlapping_constraints=False)[0]
        if not return_logits:
            masks=masks>self.mask_threshold
        return masks,output.iou_scores[0],output.pred_masks[0].clamp(-32,32)

    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=True, return_logits=False):
        """Public interface: original-image XY point coordinates and XYXY box edges."""
        import torch
        if self._features is None:
            raise ValueError('Call set_image before predicting.')
        points=labels=boxes=None
        if point_coords is not None:
            if point_labels is None:
                raise ValueError('Point labels are required.')
            coords=torch.as_tensor(point_coords,dtype=torch.float32,device=self.device)
            points=self._transforms.transform_coords(coords,normalize=True,orig_hw=self._orig_hw).reshape(1,-1,2)
            labels=torch.as_tensor(point_labels,dtype=torch.int64,device=self.device).reshape(1,-1)
        if box is not None:
            corners=torch.as_tensor(box,dtype=torch.float32,device=self.device).reshape(-1,2,2)
            boxes=self._transforms.transform_coords(corners,normalize=True,orig_hw=self._orig_hw).reshape(-1,4)
        outputs=self._predict(points,labels,boxes,multimask_output=multimask_output,return_logits=return_logits)
        return tuple(x[0].detach().cpu().numpy() for x in outputs)


def automatic_generator(predictor, settings):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    # The constructor only reads image_size from its model argument. Replace its
    # predictor immediately: the SAM 2 decoder is never loaded or called here.
    generator=SAM2AutomaticMaskGenerator(predictor,mask_threshold=0.0,use_m2m=False,
                                         multimask_output=True,**settings)
    generator.predictor=predictor
    return generator
