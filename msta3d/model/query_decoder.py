import torch
import torch.nn as nn
from msta3d.model.utils import MultiheadAttention


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_offsets, attn_masks=None, pe=None):
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask=attn_masks[i])
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        return outputs

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nheads=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nheads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, _ = self.attn(x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))
        elif activation_fn == 'gelu':
            self.net = nn.Sequential( nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output

class MLP(nn.Sequential):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_channels, out_channels))
        return super().__init__(*modules)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

class QueryDecoder(nn.Module):

    def __init__(self,
                 num_layer=1,
                 num_query=100,
                 num_class=18,
                 in_channel=32,
                 d_model=256,
                 nhead=8,
                 hidden_dim=1024,
                 dropout=0.0,
                 activation_fn='relu',
                 iter_pred=False,
                 attn_mask=False,
                 pe=False):

        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.d_model = d_model

        self.query = nn.Embedding(num_query, self.d_model)
        if pe:
            self.pe = nn.Embedding(num_query, self.d_model)

        self.input_proj = nn.Sequential(
            nn.Linear(in_channel, self.d_model), nn.LayerNorm(self.d_model), nn.ReLU())

        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(self.d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(self.d_model , nhead))
            self.ffn_layers.append(
                FFN(self.d_model, hidden_dim, dropout, activation_fn))

        self.out_norm = nn.LayerNorm(self.d_model)

        self.x_mask = MLP(in_channel, self.d_model - 6, self.d_model - 6, 2)
        self.x_box = MLP(in_channel, self.d_model, 6, 2)
        self.x_mask_box = MLP(self.d_model, self.d_model, self.d_model, 1)

        self.out_box = MLP(self.d_model, self.d_model, 6, 2)
        self.out_box_score = MLP(self.d_model, self.d_model, 1, 2)
        self.out_cls = MLP(self.d_model, self.d_model, num_class + 1, 2)
        self.out_score = MLP(self.d_model, self.d_model, 1, 2)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight.data)

    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feat)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return pred_masks, attn_masks

    def box_regularizer(self, query, pred_boxes, box_feats, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            pred_mask_box = pred_boxes[i].unsqueeze(1) - box_feats[start_id:end_id].unsqueeze(0)
            pred_mask_box = torch.cat((mask_feats[start_id:end_id].repeat(pred_mask_box.shape[0], 1, 1), pred_mask_box), dim=2)
            pred_mask_box = self.x_mask_box(pred_mask_box)
            pred_mask = torch.einsum('bnd,bmd->bnm', query[i].unsqueeze(1), pred_mask_box).squeeze(1)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_offsets, box_feats=None):
        query = self.out_norm(query)
        pred_labels = self.out_cls(query)
        pred_scores = self.out_score(query)
        if box_feats is not None:
            pred_boxes = self.out_box(query)
            pred_box_scores = self.out_box_score(query)
            pred_masks, attn_masks = self.box_regularizer(query, pred_boxes, box_feats, mask_feats, batch_offsets)
            return pred_boxes, pred_box_scores, pred_labels, pred_scores, pred_masks, attn_masks
        else:
            pred_masks, attn_masks = self.get_mask(query[:,:,:self.d_model-6], mask_feats, batch_offsets)
            return pred_labels, pred_scores, pred_masks, attn_masks

    def forward(self, x_h, x_l, batch_offsets_h, batch_offsets_l):
        B = len(batch_offsets_h) - 1
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)
        if getattr(self, 'pe', None):
            pe = self.pe.weight.unsqueeze(0).repeat(B, 1, 1)
        else:
            pe = None

        prediction_labels, prediction_masks, prediction_scores, prediction_boxes, prediction_box_scores = [], [], [], [], []
        inst_feats_h, inst_feats_l = self.input_proj(x_h), self.input_proj(x_l)
        mask_feats_h, mask_feats_l = self.x_mask(x_h), self.x_mask(x_l)
        box_feats = self.x_box(x_h)

        pred_boxes, pred_box_scores, pred_labels, pred_scores, pred_masks, attn_masks_h = self.prediction_head(query, mask_feats_h, batch_offsets_h, box_feats)
        _, _, _, attn_masks_l = self.prediction_head(query, mask_feats_l, batch_offsets_l, None)
        prediction_boxes.append(pred_boxes)

        prediction_box_scores.append(pred_box_scores)
        prediction_labels.append(pred_labels)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)

        for i in range(self.num_layer):
            query_h = self.cross_attn_layers[i](
                inst_feats_h, query, batch_offsets_h, attn_masks_h, pe)
            query_l = self.cross_attn_layers[i](
                inst_feats_l, query, batch_offsets_l, attn_masks_l, pe)
            query_h = self.self_attn_layers[i](query_h)
            query_l = self.self_attn_layers[i](query_l)
            query = query_h * query_l
            query = self.ffn_layers[i](query)

            pred_boxes, pred_box_scores, pred_labels, pred_scores, pred_masks, attn_masks_h = self.prediction_head(query, mask_feats_h, batch_offsets_h, box_feats)
            _, _, _, attn_masks_l = self.prediction_head(query, mask_feats_l, batch_offsets_l, None)

            prediction_boxes.append(pred_boxes)
            prediction_box_scores.append(pred_box_scores)
            prediction_labels.append(pred_labels)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
        return {
            'labels': pred_labels,
            'masks': pred_masks,
            'mask_scores': pred_scores,
            'boxes': pred_boxes,
            'box_scores': pred_box_scores,
            'aux_outputs': [{'labels': a, 'masks': b, 'mask_scores': c, 'boxes': d, 'box_scores': e} for a, b, c, d, e in zip(prediction_labels[:-1], prediction_masks[:-1], prediction_scores[:-1], prediction_boxes[:-1], prediction_box_scores[:-1])],
        }
