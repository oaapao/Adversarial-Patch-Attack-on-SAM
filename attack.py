from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import torch.nn.functional as F


class MaskedNegativeLoss(nn.Module):
    def __init__(self):
        super(MaskedNegativeLoss, self).__init__()

    def forward(self, output, gt_mask):
        masked_output = output * gt_mask
        # negative_output = -masked_output
        loss = torch.mean(masked_output)
        return loss


class SamPredictor:
    def __init__(
            self,
            sam_model: Sam,
    ) -> None:
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def mask_adversarial_example(self, gt_mask: torch.Tensor, 
                                 edit_mask: torch.Tensor, 
                                 image: np.ndarray, step: int,
                                 point_coords: Optional[np.ndarray] = None,
                                 point_labels: Optional[np.ndarray] = None):

        # step 1: make the input require_grad
        input_image = self.transform.apply_image(image)
        
        origin_image = torch.as_tensor(input_image, device=self.device).float().permute(2, 0, 1).contiguous()[None, :, :, :]
        p = torch.as_tensor(input_image, device=self.device).float().permute(2, 0, 1).contiguous()[None, :, :, :]
        p.requires_grad = True
        optimizer = optim.Adam([p], lr=1)
        
        loss_function = MaskedNegativeLoss()
                                   
        # step 2: process the mask of patch
        edit_mask = edit_mask.unsqueeze(1).repeat(1, 3, 1, 1).float()
        edit_mask = F.interpolate(edit_mask, size=(p.shape[-2], p.shape[-1]), mode='bilinear', align_corners=False)


        for _ in tqdm(range(step)):
            optimizer.zero_grad()
            
            clamped_p = p.clamp(0., 255.)
            # apply edit_mask
            applied_image = (clamped_p * edit_mask + origin_image * (1 - edit_mask))
            self.set_torch_image(applied_image, image.shape[:2])

            coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

            masks, _, _ = self.predict_torch(
                coords_torch,
                labels_torch,
                box_torch,
                mask_input_torch,
                multimask_output=False,
                return_logits=True,
            )

            pred_mask_logits = masks[0]
            loss = loss_function(pred_mask_logits, gt_mask)
            loss.backward()
            
            optimizer.step()

            
        clamped_p = p.clamp(0., 255.)
        applied_image = (clamped_p * edit_mask + origin_image * (1 - edit_mask))
        to_save_np = applied_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        to_save_image = Image.fromarray(np.uint8(to_save_np))
        to_save_image.save("./adversarial_example.jpg")

    def set_image(
            self,
            image: np.ndarray,
            image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    def set_torch_image(
            self,
            transformed_image: torch.Tensor,
            original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
                len(transformed_image.shape) == 4
                and transformed_image.shape[1] == 3
                and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        model_dtype = self.model.mask_decoder.iou_prediction_head.layers[0].weight.dtype
        self.features = self.model.image_encoder(input_image.to(model_dtype))
        self.is_image_set = True

    def predict(
            self,
            point_coords: Optional[np.ndarray] = None,
            point_labels: Optional[np.ndarray] = None,
            box: Optional[np.ndarray] = None,
            mask_input: Optional[np.ndarray] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                    point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0]
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def predict_torch(
            self,
            point_coords: Optional[torch.Tensor],
            point_labels: Optional[torch.Tensor],
            boxes: Optional[torch.Tensor] = None,
            mask_input: Optional[torch.Tensor] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        if low_res_masks.is_nested:

            masks = []
            for lrm, input_size, original_size in zip(low_res_masks.unbind(), self.input_sizes, self.original_sizes,
                                                      strict=True):
                # Upscale the masks to the original image resolution
                m = self.model.postprocess_masks(lrm, input_size, original_size)
                masks.append(m)
            masks = torch.nested.nested_tensor(masks, layout=torch.strided)
        else:
            # Upscale the masks to the original image resolution
            masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


if __name__ == '__main__':
    sam_checkpoint = "path/to/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image = cv2.imread('path/to/images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        return_logits=True
    )
    gt_mask = masks > 0

    predictor.mask_adversarial_example(gt_mask=gt_mask,
                                       edit_mask=gt_mask,
                                       image=image, 
                                       step=300, 
                                       point_coords=input_point, 
                                       point_labels=input_label)
