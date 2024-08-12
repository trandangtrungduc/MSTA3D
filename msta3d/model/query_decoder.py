import torch
import torch.nn as nn
from msta3d.model.utils import MLP, FeedForwardLayer, CrossAttentionLayer, SelfAttentionLayer

class TwinAttentionDecoder(nn.Module):

    def __init__(self,
                 num_layer = 6,
                 num_hidden_layer = 2,
                 num_query = 200,
                 box_query = True,
                 merge_query = True,
                 num_class = 18,
                 d_b = 32,
                 d_o = 256,
                 num_head = 8,
                 hidden_dim = 1024,
                 dropout = 0.0,
                 activation_fn = 'relu',
                 iter_pred = False,
                 pe = False
                 ):

        super().__init__()
        self.d_o = d_o
        self.d_s = self.d_o - 6
        self.num_layer = num_layer
        self.num_query = num_query
        self.box_query = box_query
        self.merge_query = merge_query
        self.iter_pred = iter_pred

        self.query = nn.Embedding(num_query, self.d_o)
        if pe:
            self.pe = nn.Embedding(num_query, self.d_o)
        self.input_proj = nn.Sequential(
            nn.Linear(d_b, self.d_o), nn.LayerNorm(self.d_o), nn.ReLU())

        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(self.d_o, num_head, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(self.d_o, num_head))
            self.ffn_layers.append(FeedForwardLayer(self.d_o, hidden_dim, dropout, activation_fn))

        self.out_norm = nn.LayerNorm(self.d_o)

        self.x_mask = MLP(d_b, self.d_s, self.d_s, num_hidden_layer)

        if self.box_query:
            self.x_box = MLP(d_b, self.d_o, self.d_o - self.d_s, num_hidden_layer)
            self.x_mask_box = MLP(self.d_o, self.d_o, self.d_o, 1)

            self.out_box_score = MLP(self.d_o, self.d_o, 1, num_hidden_layer)
            if self.merge_query:
                self.out_box = MLP(self.d_o, self.d_o, self.d_o - self.d_s, num_hidden_layer)
            else:
                self.out_box = MLP(self.d_o - self.d_s, self.d_o, self.d_o - self.d_s, num_hidden_layer)

        self.out_cls = MLP(self.d_o, self.d_o, num_class + 1, num_hidden_layer)
        self.out_score = MLP(self.d_o, self.d_o, 1, num_hidden_layer)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):

        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight.data)

    def no_box_regularizer(self,
                           query,
                           mask_feats,
                           batch_offsets
                 ):

        pred_masks = []
        attn_masks = []

        for i in range(len(batch_offsets) - 1):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]

            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd, md -> nm', query[i], mask_feat)

            attn_mask = (pred_mask.sigmoid() < 0.5).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_mask = attn_mask.detach()

            attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def box_regularizer(self,
                        query,
                        pred_boxes,
                        box_feats,
                        mask_feats,
                        batch_offsets
                        ):

        batch_size = len(batch_offsets)

        pred_masks = []
        attn_masks = []

        for i in range(batch_size - 1):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]

            pred_mask_box = pred_boxes[i].unsqueeze(1) - box_feats[start_id:end_id].unsqueeze(0)
            pred_mask_box = torch.cat((mask_feats[start_id:end_id].repeat(pred_mask_box.shape[0], 1, 1), pred_mask_box), dim=2)
            pred_mask_box = self.x_mask_box(pred_mask_box)

            pred_mask = torch.einsum('bnd, bmd -> bnm', query[i].unsqueeze(1), pred_mask_box).squeeze(1)

            attn_mask = (pred_mask.sigmoid() < 0.5).bool()
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_mask = attn_mask.detach()

            attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def prediction_head(self,
                        query,
                        mask_feats,
                        batch_offsets,
                        box_feats
                        ):

        query = self.out_norm(query)
        pred_labels = self.out_cls(query)
        pred_scores = self.out_score(query)

        if box_feats is not None:
            pred_box_scores = self.out_box_score(query)
            if self.merge_query:
                pred_boxes = self.out_box(query)
            else:
                box_query = query[:, :, self.d_s:]
                # pred_boxes = box_query
                pred_boxes = self.out_box(box_query)

            pred_masks, attn_masks = self.box_regularizer(query, pred_boxes, box_feats, mask_feats, batch_offsets)
            return pred_boxes, pred_box_scores, pred_labels, pred_scores, pred_masks, attn_masks
        else:
            sem_query = query[:, :, :self.d_s]

            pred_masks, attn_masks = self.no_box_regularizer(sem_query, mask_feats, batch_offsets)
            return pred_labels, pred_scores, pred_masks, attn_masks

    def forward(self,
                x_h,
                x_l,
                batch_offsets_h,
                batch_offsets_l
                ):

        batch_size = len(batch_offsets_h) - 1

        prediction_masks = []
        prediction_labels = []
        prediction_scores = []

        mask_feats_h = self.x_mask(x_h)
        mask_feats_l = self.x_mask(x_l)
        inst_feats_h = self.input_proj(x_h)
        inst_feats_l = self.input_proj(x_l)

        if getattr(self, 'pe', None):
            pe_weight = self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            pe_weight = None
        pe = pe_weight

        query_weight = self.query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.box_query:
            box_feats = self.x_box(x_h)

            sem_query = query_weight[:, :, :self.d_s]
            box_query = query_weight[:, :, self.d_s:]
            query = torch.cat((sem_query, box_query), dim=-1)

            pred_boxes, pred_box_scores, pred_labels, pred_scores, pred_masks, attn_masks_h = self.prediction_head(query, mask_feats_h, batch_offsets_h, box_feats)

            prediction_boxes = []
            prediction_box_scores = []
            prediction_boxes.append(pred_boxes)
            prediction_box_scores.append(pred_box_scores)
        else:
            box_feats = None
            query = query_weight

            pred_labels, pred_scores, pred_masks, attn_masks_h = self.prediction_head(query, mask_feats_h, batch_offsets_h, box_feats)

        _, _, _, attn_masks_l = self.prediction_head(query, mask_feats_l, batch_offsets_l, None)

        prediction_masks.append(pred_masks)
        prediction_labels.append(pred_labels)
        prediction_scores.append(pred_scores)

        for i in range(self.num_layer):
            query_h = self.cross_attn_layers[i](inst_feats_h, query, batch_offsets_h, attn_masks_h, pe)
            query_l = self.cross_attn_layers[i](inst_feats_l, query, batch_offsets_l, attn_masks_l, pe)

            query_h = self.self_attn_layers[i](query_h)
            query_l = self.self_attn_layers[i](query_l)

            query = query_h * query_l
            query = self.ffn_layers[i](query)

            if self.box_query:
                pred_boxes, pred_box_scores, pred_labels, pred_scores, pred_masks, attn_masks_h = self.prediction_head(query, mask_feats_h, batch_offsets_h, box_feats)
                prediction_boxes.append(pred_boxes)
                prediction_box_scores.append(pred_box_scores)
            else:
                pred_labels, pred_scores, pred_masks, attn_masks_h = self.prediction_head(query, mask_feats_h, batch_offsets_h, box_feats)

            _, _, _, attn_masks_l = self.prediction_head(query, mask_feats_l, batch_offsets_l, None)

            prediction_masks.append(pred_masks)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)

        if self.box_query:
            return {
                'labels': pred_labels,
                'masks': pred_masks,
                'mask_scores': pred_scores,
                'boxes': pred_boxes,
                'box_scores': pred_box_scores,
                'aux_outputs': [{'labels': a, 'masks': b, 'mask_scores': c, 'boxes': d, 'box_scores': e} for a, b, c, d, e in zip(prediction_labels[:-1], prediction_masks[:-1], prediction_scores[:-1], prediction_boxes[:-1], prediction_box_scores[:-1])]
            }
        else:
            return {
                'labels': pred_labels,
                'masks': pred_masks,
                'mask_scores': pred_scores,
                'aux_outputs': [{'labels': a, 'masks': b, 'mask_scores': c} for a, b, c in zip(prediction_labels[:-1], prediction_masks[:-1], prediction_scores[:-1])]
            }