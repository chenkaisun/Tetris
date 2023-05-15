import gc
import math
import random
import warnings
from typing import Optional, Tuple
from IPython import embed
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from model.activations import ACT2FN
from model.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from model.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputCustom,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
# from model.modeling_utils import PreTrainedModel
# from .utils import logging
# from bart.configuration_bart import BartConfig
# from cse import EventReasoningModule
# from model_utils import get_tensor_info
# from .deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
# from model.gnn import GNN
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartClassificationHead, BartForConditionalGeneration


class CoherenceClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            pooler_dropout,
    ):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, inner_dim), nn.ReLU())

        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    def forward(self, encoder_hidden_states, encoder_mask, decoder_hidden_states, decoder_mask):
        """
        encoder_hidden_states / decoder_hidden_states: (bsz, len, dim)
        encoder_mask / decoder_mask: (bsz, len)
        """
        # # use MLP
        proj_dec_h = self.projection(decoder_hidden_states) * decoder_mask.unsqueeze(-1)
        avg_abs = self.avg_pool(proj_dec_h, decoder_mask)
        combine = torch.sigmoid(self.out_proj(avg_abs))
        return combine


class MarginRankingLoss(torch.nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, p_scores, n_scores, weights=None):
        scores = self.margin - p_scores + n_scores
        scores = scores.clamp(min=0)
        if weights is not None:
            scores = weights * scores
        return scores


# def coh_loss1():
#     coh_loss


class ContrastiveLearner(nn.Module):
    def __init__(self):
        super(ContrastiveLearner, self).__init__()
        self.criterion = MarginRankingLoss(0.5)


def get_contrastive_loss(self, pos_scores, neg_scores, weights=None):
    return self.criterion(pos_scores, neg_scores, weights)


class ContrastiveLearningHead(nn.Module):
    """Head for contrastive learning tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
    ):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, inner_dim), nn.ReLU())
        self.cos = nn.CosineSimilarity(dim=-1)

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    def forward(self, encoder_hidden_states, encoder_mask, decoder_hidden_states, decoder_mask):
        """
        encoder_hidden_states / decoder_hidden_states: (bsz, len, dim)
        encoder_mask / decoder_mask: (bsz, len)
        """
        # use distance
        proj_enc_h = self.projection(encoder_hidden_states) * encoder_mask.unsqueeze(-1)
        proj_dec_h = self.projection(decoder_hidden_states) * decoder_mask.unsqueeze(-1)

        avg_doc = self.avg_pool(proj_enc_h, encoder_mask)
        avg_abs = self.avg_pool(proj_dec_h, decoder_mask)
        combine = self.cos(avg_doc, avg_abs).unsqueeze(-1)
        return combine


class ScriptLearner(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def add_modules(self, args):
        self.use_contrastive = args.do_cl
        self.fill_empty_neg_with_null_str = args.fill_empty_neg_with_null_str
        self.orig_cl_way = args.orig_cl_way
        # self.cl_sample_types = [int(item.strip()) for item in args.cl_sample_types.strip().split()]#set()

        self.cl_sample_types = [int(item.strip()) for item in list(args.cl_sample_types.strip())]  # set()

        self.cl_empty_mode = int(args.cl_empty_mode)

        self.tokenizer = args.tokenizer
        if self.use_contrastive:
            dim_size = args.plm_hidden_dim
            self.coherence_classifer = CoherenceClassificationHead(
                input_dim=dim_size,
                inner_dim=dim_size,
                num_classes=1,
                pooler_dropout=0.1,
            )
            self.coherence_classifer.apply(self._init_weights)

            # self.contrastive_head = ContrastiveLearningHead(
            #     input_dim=dim_size,
            #     inner_dim=dim_size,
            # )
            # self.contrastive_head.apply(self._init_weights)

        # self.cross_entropy = CrossEntropyLoss(reduction="sum")
        self.cl_type = args.cl_type
        self.cl_weight = args.cl_weight

    def get_contrastive_info(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_eval=None
    ):

        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids,
                                  decoder_attention_mask=decoder_attention_mask,
                                  head_mask=head_mask,
                                  decoder_head_mask=decoder_head_mask,
                                  cross_attn_head_mask=cross_attn_head_mask,
                                  encoder_outputs=encoder_outputs,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  decoder_inputs_embeds=decoder_inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  output_hidden_states=is_eval is not None,
                                  return_dict=return_dict)
        if is_eval is not None:
            # loss, logits = outputs.loss, outputs.logits
            last_decoder_hidden_states = outputs["decoder_hidden_states"][-1]
            last_encoder_hidden_states = outputs["encoder_hidden_states"][-1]  # (batch_size, sequence_length, hidden_size)

            decoder_label_mask = labels != -100  # self.tokenizer.pad_token_id
            coherence_scores = self.coherence_classifer(
                encoder_hidden_states=last_encoder_hidden_states,
                encoder_mask=attention_mask,
                decoder_hidden_states=last_decoder_hidden_states,
                decoder_mask=decoder_label_mask,
            )
            # coherence_scores_list.append(coherence_scores)
            outputs["coherence_scores"] = coherence_scores

            # decoder_label_mask = labels != -100  # self.tokenizer.pad_token_id
            # contrastive_scores = self.contrastive_head(
            #     encoder_hidden_states=last_encoder_hidden_states,
            #     encoder_mask=attention_mask,
            #     decoder_hidden_states=last_decoder_hidden_states,
            #     decoder_mask=decoder_label_mask,
            # )
            # # coherence_scores_list.append(contrastive_scores)
            # outputs["probability_scores"] = contrastive_scores
            #
            # outputs["loss"] = loss

        # outputs["loss"] = loss
        return outputs

        # outputs["probability_scores"] = contrastive_scores

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # neg_samples_decoder_input_ids_list=None,
            # neg_samples_decoder_attention_mask_list=None,
            # neg_samples_labels_list=None,
            neg_samples=None,
            is_eval=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """

        output = self.get_contrastive_info(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           decoder_input_ids=decoder_input_ids,
                                           decoder_attention_mask=decoder_attention_mask,
                                           head_mask=head_mask,
                                           decoder_head_mask=decoder_head_mask,
                                           cross_attn_head_mask=cross_attn_head_mask,
                                           encoder_outputs=encoder_outputs,
                                           past_key_values=past_key_values,
                                           inputs_embeds=inputs_embeds,
                                           decoder_inputs_embeds=decoder_inputs_embeds,
                                           labels=labels,
                                           use_cache=use_cache,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states,
                                           return_dict=return_dict,
                                           is_eval=is_eval)

        if self.use_contrastive and is_eval is not None:
            neg_samples_decoder_input_ids_list = []
            neg_samples_decoder_attention_mask_list = []
            neg_samples_labels_list = []
            neg_samples_mask_list = []
            for neg_samp in neg_samples:
                neg_samples_decoder_input_ids_list.append(neg_samp["decoder_input_ids"])
                neg_samples_decoder_attention_mask_list.append(neg_samp["decoder_attention_mask"])
                neg_samples_labels_list.append(neg_samp["labels"])
                neg_samples_mask_list.append(neg_samp["cl_mask"])

            coh_loss_list = []
            # margin_loss_list = []
            coh_score_list = [output["coherence_scores"]]
            # coh_target_list = [torch.ones(output["coherence_scores"].shape).long().to("cuda")]
            coh_target_list = [torch.ones(output["coherence_scores"].shape, dtype=output["coherence_scores"].dtype).to("cuda")]
            coh_contrastive_loss = None

            # for cl_sample_type_id in self.cl_sample_types:

            loop_idxs= self.cl_sample_types if self.orig_cl_way else range(len(neg_samples_decoder_input_ids_list))

            for cl_sample_type_id in loop_idxs:#len(neg_samples_decoder_input_ids_list)
                neg_samples_labels = neg_samples_labels_list[cl_sample_type_id]
                neg_samples_mask = neg_samples_mask_list[cl_sample_type_id]

                """currently decoder attn mask is None"""
                neg_samples_labels = neg_samples_labels.to(self.device)
                neg_samples_mask = neg_samples_mask.to(self.device)

                neg_outputs = self.get_contrastive_info(input_ids=input_ids,
                                                        attention_mask=attention_mask,
                                                        decoder_input_ids=None,
                                                        decoder_attention_mask=None,
                                                        head_mask=head_mask,
                                                        decoder_head_mask=decoder_head_mask,
                                                        cross_attn_head_mask=cross_attn_head_mask,
                                                        encoder_outputs=encoder_outputs,
                                                        past_key_values=past_key_values,
                                                        inputs_embeds=inputs_embeds,
                                                        decoder_inputs_embeds=decoder_inputs_embeds,
                                                        labels=neg_samples_labels,
                                                        use_cache=use_cache,
                                                        output_attentions=output_attentions,
                                                        output_hidden_states=output_hidden_states,
                                                        return_dict=return_dict,
                                                        is_eval=is_eval)

                pos_coh_scores = output["coherence_scores"]
                neg_coh_scores = neg_outputs["coherence_scores"]
                # print("pos: ", pos_coh_scores)
                # print("neg: ", neg_coh_scores)

                cur_coh_loss = MarginRankingLoss(0.5)(pos_coh_scores, neg_coh_scores)

                tmp_score = cur_coh_loss if self.cl_empty_mode == 0 else cur_coh_loss * neg_samples_mask
                coh_loss_list.append(tmp_score)
                coh_score_list.append(neg_coh_scores)
                coh_target_list.append(torch.zeros(neg_coh_scores.shape).long().to("cuda"))

                # outputs["probability_scores"] = contrastive_scores
            if self.cl_type == 1:
                # print(coh_loss_list)
                if self.cl_empty_mode == 0:
                    coh_contrastive_loss = torch.mean(torch.stack(coh_loss_list))
                else:
                    if self.fill_empty_neg_with_null_str:
                        coh_contrastive_loss = torch.sum(torch.stack(coh_loss_list)) / torch.sum(torch.stack(neg_samples_mask_list))
                    else:
                        sum_masks=torch.sum(torch.stack(neg_samples_mask_list))
                        if torch.is_nonzero(sum_masks):
                            coh_contrastive_loss = torch.sum(torch.stack(coh_loss_list)) / torch.sum(torch.stack(neg_samples_mask_list))
                        else:
                            coh_contrastive_loss = torch.sum(torch.stack(coh_loss_list))*torch.tensor(0.0).to("cuda")
                # print("coh_loss: ", coh_contrastive_loss)
            else:
                all_coh_scores = torch.cat(coh_score_list, 1)
                all_target = torch.cat(coh_target_list, 1)
                all_coh_target = torch.zeros(all_coh_scores.size()[0]).long().to("cuda")
                # print("coh_score_list: ", coh_score_list)
                # print("coh_target_list: ", coh_target_list)
                # print("all_coh_scores: ", all_coh_scores)
                # print("all_target: ", all_target)

                if self.cl_type == 2:
                    all_coh_label = torch.argmax(all_target, dim=1)
                    # print("all_coh_label: ", all_coh_label)
                    coh_contrastive_loss = nn.CrossEntropyLoss(reduction="none")(all_coh_scores, all_coh_label)
                    # print("ce loss: ", coh_contrastive_loss)
                else:
                    contrastive_norminator = (all_coh_scores.exp() * all_target).sum(1)
                    contrastive_denorminator = all_coh_scores.exp().sum(1)
                    coh_contrastive_loss = contrastive_norminator / contrastive_denorminator
                    coh_contrastive_loss = - coh_contrastive_loss.log()
                    coh_contrastive_loss = coh_contrastive_loss.mean()
                    # print("coh_contrastive_loss: ", coh_contrastive_loss)
            output["loss"] += self.cl_weight * coh_contrastive_loss

            # # compute loss
            # lm_dist = F.softmax(output.logits, dim=-1)
            # vocab_size = output.logits.shape[-1]
            # flat_target = labels.view(-1, 1)
            # flat_log_probs = lm_dist.log().view(-1, vocab_size)
            # aa=torch.where(flat_target == -100, torch.tensor(1).to(self.device), flat_target)
            # ce = flat_log_probs.gather(index=aa, dim=-1)
            # ce = ce[aa != self.tokenizer.pad_token_id]
            # # ce = ce[flat_target != -100]
            # loss = -1 * ce.mean()
            # print("loss", loss)
            # print("output.loss", output.loss)

        return output

        #
        # if labels is not None:
        #     if decoder_input_ids is None:
        #         decoder_input_ids = shift_tokens_right(
        #             labels, self.config.pad_token_id, self.config.decoder_start_token_id
        #         )
