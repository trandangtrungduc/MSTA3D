import time
import torch
import torch.nn as nn

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

def box_regularizer(query, pred_boxes, box_feats, mask_feats, batch_offsets):
        pred_masks = []
        # loop over batch size
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            pred_mask_box = pred_boxes[i].unsqueeze(1) - box_feats[start_id:end_id].unsqueeze(0)
            pred_mask_box = torch.cat((mask_feats[start_id:end_id].repeat(pred_mask_box.shape[0], 1, 1), pred_mask_box), dim=2)
            # pred_mask_box = x_mask_box(pred_mask_box) # temporary remove linear layer for checking code
            pred_mask = torch.einsum('bnd,bmd->bnm', query[i].unsqueeze(1), pred_mask_box).squeeze(1)
            pred_masks.append(pred_mask)
        return pred_masks

def box_regularizer_loop(query, pred_boxes, box_feats, mask_feats, batch_offsets):
    batch_size, num_query, d_model = query.shape[0], query.shape[1], query.shape[2]
    pred_masks = []
    # loop over batch size
    for i in range(len(batch_offsets) - 1):
        start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
        pred_mask_box = pred_boxes[i].unsqueeze(1) - box_feats[start_id:end_id].unsqueeze(0)
        num_batch_point = pred_mask_box.shape[1]
        # loop over query
        pred_mask = torch.zeros((num_query, num_batch_point))
        for query_id in range(num_query):
            pred_spp_mask_box = torch.zeros((num_batch_point, d_model))
            # loop over superpoint-wise features
            for spp_id in range(num_batch_point):
                pred_spp_mask_box[spp_id] = torch.cat((mask_feats[start_id:end_id][spp_id], pred_mask_box[query_id][spp_id]))
            pred_mask[query_id] = query[i][query_id] @ pred_spp_mask_box.T
            # Add linear layer here if necessary
            #
        pred_masks.append(pred_mask)
    return pred_masks

if __name__ == '__main__':
    d_model = 14
    d_box = 6
    d_mask = 8
    num_point = 10
    num_query  = 5
    batch_size = 2

    mask_feats = torch.rand(num_point, d_mask)
    box_feats = torch.rand(num_point, d_box)
    query = torch.rand(batch_size, num_query, d_model)
    pred_boxes = torch.rand(batch_size, num_query, d_box)
    batch_offsets = torch.tensor([0, 4, 10])
    # Linear layer
    x_mask_box = MLP(d_model, d_model, d_model, 1)

    # Using broadcasting tensor
    start_1 = time.time()
    result_1 = box_regularizer(query, pred_boxes, box_feats, mask_feats, batch_offsets)
    end_1 = time.time()
    time_1 = end_1 - start_1

    # Using for loop
    start_2 = time.time()
    result_2 = box_regularizer_loop(query, pred_boxes, box_feats, mask_feats, batch_offsets)
    end_2 = time.time()
    time_2 = end_2 - start_2

    for i in range(batch_size):
        print(result_1[i])
        print(result_2[i])
        print(torch.allclose(result_1[i], result_2[i]))

    print("Calculating Time")
    print("Broadcasting: ", time_1)
    print("For Loop: ", time_2)
    print("Deviation: ", time_2/time_1)