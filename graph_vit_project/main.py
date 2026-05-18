import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append('/home/qiannnhui/GraphCL/unsupervised_TU')
from torch_geometric.loader import DataLoader
import torch_geometric
from torch_geometric.utils import degree, add_self_loops, subgraph
from aug import TUDataset_aug as TUDataset
from evaluate_embedding import evaluate_embedding

from .models import DualEncoder

def get_rw_features(edge_index, num_nodes, batch, walk_length=16):
    """
    Computes Random Walk features per graph to avoid OOM on large batches.
    """
    edge_index_sl, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    rw_list = []
    unique_batch = torch.unique(batch)
    for i in unique_batch:
        mask = (batch == i)
        node_idx = torch.where(mask)[0]
        sub_num_nodes = len(node_idx)
        sub_edge_index, _ = subgraph(node_idx, edge_index_sl, relabel_nodes=True)
        row, col = sub_edge_index
        sub_deg = degree(row, sub_num_nodes)
        sub_deg_inv = 1.0 / sub_deg
        sub_deg_inv[sub_deg_inv == float('inf')] = 0
        adj = torch.zeros(sub_num_nodes, sub_num_nodes, device=edge_index.device)
        adj[row, col] = sub_deg_inv[row]
        pe = []
        out = adj
        for _ in range(walk_length):
            pe.append(torch.diag(out))
            out = out @ adj
        rw_list.append(torch.stack(pe, dim=1))
    return torch.cat(rw_list, dim=0)

def compute_rbo(sim_matrix, p=0.98):
    B = sim_matrix.size(0)
    device = sim_matrix.device
    k = min(20, B)
    _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)
    weights = torch.pow(p, torch.arange(k, device=device).float())
    presence = torch.zeros((B, B), device=device)
    rows = torch.arange(B, device=device).unsqueeze(1).expand(-1, k)
    presence[rows, top_k_indices] = weights
    rbo_matrix = torch.mm(presence, presence.T)
    norm_factor = torch.sqrt(torch.diag(rbo_matrix).unsqueeze(0) * torch.diag(rbo_matrix).unsqueeze(1))
    rbo_matrix = rbo_matrix / (norm_factor + 1e-8)
    return rbo_matrix

def model_eval(model, dataloader, device, eval_mode, rw_length, max_nodes=1024):
    model.eval()
    x, y = [], []
    with torch.no_grad():
        for data_batch in dataloader:
            if isinstance(data_batch, (list, tuple)):
                data = data_batch[0].to(device)
            else:
                data = data_batch.to(device)
            rw_x = get_rw_features(data.edge_index, data.num_nodes, data.batch, walk_length=rw_length)
            data.rw_x = rw_x
            z_nodes, z_graph, _, _, _ = model(data, max_nodes=max_nodes)
            
            if eval_mode == 'simclr':
                emb = z_graph
            elif eval_mode == 'groupvit':
                emb = z_nodes
            else: # concat
                emb = torch.cat([z_nodes, z_graph], dim=1)
                
            x.append(emb.cpu().numpy())
            y.append(data.y.cpu().numpy())
    
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return evaluate_embedding(x, y)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_mode', type=str, default='concat', choices=['concat', 'simclr', 'groupvit'])
    parser.add_argument('--rw_length', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--active_threshold', type=float, default=0.1)
    parser.add_argument('--sim_threshold', type=float, default=0.90)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--fn_mode', type=str, default='causal_rbo', choices=['sim', 'rbo', 'causal_sim', 'causal_rbo'])
    parser.add_argument('--causal_k', type=int, default=2, help='Top K components to form causal subgraph representation')
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rbo_threshold', type=float, default=0.6)
    parser.add_argument('--rbo_p', type=float, default=0.98)
    parser.add_argument('--tensorboard_dir', type=str, default='runs/graphvit_experiment')
    parser.add_argument('--attn_mode', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--max_nodes', type=int, default=1024, help='Max nodes per graph to prevent OOM')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU DataParallel training')
    
    # Cyclic Training / Distillation arguments
    parser.add_argument('--distill', action='store_true', help='Enable teacher-student distillation')
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--distill_temp', type=float, default=0.2)
    parser.add_argument('--num_cycles', type=int, default=1)
    parser.add_argument('--distill_epochs_per_cycle', type=int, default=20)
    parser.add_argument('--cl_epochs_per_cycle', type=int, default=20)
    
    args = parser.parse_args()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.tensorboard_dir, args.dataset, current_time)
    writer = SummaryWriter(log_dir)

    # Automatically adjust batch size and max_nodes for very large datasets to avoid OOM
    if args.dataset in ['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']:
        original_bs = args.batch_size
        if args.dataset == 'DD':
            args.batch_size = min(args.batch_size, 32)
            args.max_nodes = min(args.max_nodes, 512)
        elif args.dataset.startswith('REDDIT'):
            args.batch_size = min(args.batch_size, 32)
            args.max_nodes = min(args.max_nodes, 512)
        elif args.dataset == 'COLLAB':
            args.batch_size = min(args.batch_size, 64)
            args.max_nodes = min(args.max_nodes, 512)
        print(f"--- [Auto OOM Protection] Dataset {args.dataset} detected. Batch size reduced from {original_bs} to {args.batch_size}, Max Nodes capped at {args.max_nodes} ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join('/home/qiannnhui/GraphCL/unsupervised_TU/data', args.dataset)
    
    # [ROBUST] Always use augmented data for training to ensure feature quality
    dataset_aug = TUDataset(path, name=args.dataset, aug='random4').shuffle()
    dataloader_aug = DataLoader(dataset_aug, batch_size=args.batch_size, shuffle=True)
    
    dataset_eval = TUDataset(path, name=args.dataset, aug='none').shuffle()
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False)
    
    try:
        dataset_num_features = dataset_aug.get_num_feature() if hasattr(dataset_aug, 'get_num_feature') else 1
    except:
        dataset_num_features = 1

    model = DualEncoder(dataset_num_features, hidden_dim=args.hidden_dim, rw_dim=args.rw_length).to(device)
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        from torch_geometric.nn import DataParallel
        print(f"--- Wrapping model with PyG DataParallel across {torch.cuda.device_count()} GPUs ---")
        model = DataParallel(model)
        
    base_model = model.module if isinstance(model, torch_geometric.nn.DataParallel) else model
    
    teacher = None
    if args.distill:
        from gin import Encoder as TeacherEncoder
        teacher = TeacherEncoder(dataset_num_features, 32, 5).to(device)
        teacher_path = f"/home/qiannnhui/GraphCL/unsupervised_TU/result/GCL/mode_normal/DS_{args.dataset}/aug_dnodes/Renormalization_True/RBO_anchor_True/ckpts.pth.tar"
        if os.path.exists(teacher_path):
            print(f"Loading Teacher from: {teacher_path}")
            checkpoint = torch.load(teacher_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'): new_state_dict[k[8:]] = v
                elif not k.startswith('proj_head.'): new_state_dict[k] = v
            teacher.load_state_dict(new_state_dict, strict=False)
            teacher.eval()
            for param in teacher.parameters(): param.requires_grad = False
        else:
            print(f"WARNING: Teacher checkpoint NOT found.")
            args.distill = False

    teacher_proj = torch.nn.Linear(160, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(teacher_proj.parameters()), lr=args.lr)
    total_epochs = args.num_cycles * (args.distill_epochs_per_cycle + args.cl_epochs_per_cycle)

    for epoch in range(1, total_epochs + 1):
        cycle_len = args.distill_epochs_per_cycle + args.cl_epochs_per_cycle
        rel_epoch = (epoch - 1) % cycle_len
        cycle_idx = (epoch - 1) // cycle_len
        phase = 'DISTILL' if rel_epoch < args.distill_epochs_per_cycle else 'FN_CL'
        
        # 🚀 [LR Warmup for DISTILL Phase in Cycle 2+]
        current_lr = args.lr
        if phase == 'DISTILL' and cycle_idx > 0 and rel_epoch < 3:
            # Warm up learning rate linearly from 10% to 100% over the first 3 epochs of distillation phase
            factor = (rel_epoch + 1) / 3.0
            current_lr = args.lr * factor
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        if phase == 'DISTILL':
            base_model.node_transformer.requires_grad_(True)
            base_model.graph_encoder.requires_grad_(False)
            teacher_proj.requires_grad_(True)
        else:
            base_model.node_transformer.requires_grad_(False)
            base_model.graph_encoder.requires_grad_(True)
            teacher_proj.requires_grad_(False)

        model.train()
        loss_all = 0
        epoch_tp, epoch_fp, epoch_fn = 0, 0, 0
        epoch_predicted_fn = 0
        epoch_candidate_pairs = 0
        epoch_causal_sim = 0
        total_graphs = 0
        
        for data_batch in dataloader_aug:
            optimizer.zero_grad()
            data1, data2 = data_batch
            data1, data2 = data1.to(device), data2.to(device)
            
            data1.rw_x = get_rw_features(data1.edge_index, data1.num_nodes, data1.batch, walk_length=args.rw_length)
            data2.rw_x = get_rw_features(data2.edge_index, data2.num_nodes, data2.batch, walk_length=args.rw_length)
            
            z_nodes1, z_graph1, components1, attn_list1, rw_dense1 = model(data1)
            z_nodes2, z_graph2, components2, attn_list2, rw_dense2 = model(data2)
            
            B = z_nodes1.shape[0]
            z_nodes1_norm = F.normalize(z_nodes1, dim=1)
            z_nodes2_norm = F.normalize(z_nodes2, dim=1)

            if phase == 'DISTILL':
                # 1. Self-Contrast (SimCLR style) for robustness
                logits_self = (1.0 / 0.07) * z_nodes1_norm @ z_nodes2_norm.T
                loss_self = F.cross_entropy(logits_self, torch.arange(B, device=device))
                
                loss = loss_self
                if args.distill and teacher is not None:
                    # 2. Teacher-Alignment (Direct Feature Copy)
                    with torch.no_grad():
                        if cycle_idx == 0:
                            z_teacher, _ = teacher(data1.x, data1.edge_index, data1.batch)
                            z_teacher_proj = F.normalize(teacher_proj(z_teacher), dim=1)
                        else:
                            # From Cycle 2 onwards, use the newly trained graph_encoder as Teacher!
                            z_teacher, _ = base_model.graph_encoder(data1.x, data1.edge_index, data1.batch)
                            z_teacher_proj = F.normalize(base_model.graph_proj(z_teacher), dim=1)
                    
                    logits_to_teacher = (1.0 / 0.07) * z_nodes1_norm @ z_teacher_proj.T
                    loss_to_teacher = F.cross_entropy(logits_to_teacher, torch.arange(B, device=device))
                    
                    # 🚀 [Softer Distillation Temperature for Cycle 2+ to prevent Collapse]
                    current_temp = args.distill_temp if cycle_idx == 0 else min(1.0, args.distill_temp * 2.0)
                    sim_T = torch.softmax(z_teacher_proj @ z_teacher_proj.T / current_temp, dim=-1)
                    sim_S = torch.log_softmax(z_nodes1_norm @ z_nodes1_norm.T / current_temp, dim=-1)
                    loss_distill = F.kl_div(sim_S, sim_T, reduction='batchmean')
                    
                    loss = loss + loss_to_teacher + args.distill_weight * loss_distill
                    writer.add_scalar('Loss/distill_kl', loss_distill.item(), epoch)
                    writer.add_scalar('Loss/cl_to_teacher', loss_to_teacher.item(), epoch)
                writer.add_scalar('Loss/cl_self', loss_self.item(), epoch)
            else:
                # FN_CL phase: Train GCL using Student's FN Mask
                fn_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
                if len(attn_list1) > 0:
                    mode = args.attn_mode
                    attn = attn_list1[-1][mode]
                    if attn.dim() == 4: attn = attn.mean(dim=1)
                    
                    if 'sim' in args.fn_mode:
                        comp_weight = attn.sum(dim=2)
                        features = components1 - components1.mean(dim=-1, keepdim=True)
                        features = F.normalize(features, dim=-1)
                    elif 'rbo' in args.fn_mode:
                        A0 = attn_list1[0][mode]
                        if A0.dim() == 4: A0 = A0.mean(dim=1)
                        if len(attn_list1) > 1:
                            A1 = attn_list1[1][mode]
                            if A1.dim() == 4: A1 = A1.mean(dim=1)
                            A_comb = A1 @ A0
                        else: A_comb = A0
                        comp_weight = A_comb.sum(dim=2)
                        fingerprint = A_comb @ rw_dense1
                        features = fingerprint - fingerprint.mean(dim=-1, keepdim=True)
                        features = F.normalize(features, dim=-1)
                    
                    if args.fn_mode.startswith('causal_'):
                        k = min(args.causal_k, comp_weight.size(1))
                        topk_idx = torch.topk(comp_weight, k, dim=1).indices
                        topk_features = torch.gather(features, 1, topk_idx.unsqueeze(-1).expand(-1, -1, features.size(-1)))
                        causal_rep = topk_features.mean(dim=1)
                        causal_rep = F.normalize(causal_rep, dim=-1)
                        sim_matrix = causal_rep @ causal_rep.T
                        match_condition = sim_matrix > args.sim_threshold
                    else:
                        all_pairs_sim = torch.einsum('ikd,jld->ijkl', features, features)
                        mask_active = (comp_weight > args.active_threshold).float()
                        sim_for_max_i = all_pairs_sim + (1.0 - mask_active.view(1, B, 1, 4)) * -2.0
                        best_match_i_in_j = sim_for_max_i.max(dim=3).values
                        sim_for_max_j = all_pairs_sim + (1.0 - mask_active.view(B, 1, 4, 1)) * -2.0
                        best_match_j_in_i = sim_for_max_j.max(dim=2).values
                        mean_match_i = (best_match_i_in_j * mask_active.view(B, 1, 4)).sum(dim=2) / (mask_active.view(B, 1, 4).sum(dim=2) + 1e-8)
                        mean_match_j = (best_match_j_in_i * mask_active.view(1, B, 4)).sum(dim=2) / (mask_active.view(1, B, 4).sum(dim=2) + 1e-8)
                        match_condition = (mean_match_i > args.sim_threshold) & (mean_match_j > args.sim_threshold)
                        
                    if 'rbo' in args.fn_mode:
                        z_graph_norm = F.normalize(z_graph1, dim=-1)
                        rbo_matrix = compute_rbo(z_graph_norm @ z_graph_norm.T, p=args.rbo_p)
                        fn_mask = match_condition & (rbo_matrix > args.rbo_threshold)
                    else:
                        fn_mask = match_condition
                    fn_mask.fill_diagonal_(False)

                true_fn_mask = (data1.y.unsqueeze(0) == data1.y.unsqueeze(1))
                true_fn_mask.fill_diagonal_(False)
                epoch_tp += (fn_mask & true_fn_mask).sum().item()
                epoch_fp += (fn_mask & ~true_fn_mask).sum().item()
                epoch_fn += (~fn_mask & true_fn_mask).sum().item()

                # 🚀 Track predicted False Negatives and similarity for diagnosis
                epoch_predicted_fn += fn_mask.sum().item()
                epoch_candidate_pairs += B * (B - 1)
                
                if len(attn_list1) > 0:
                    if args.fn_mode.startswith('causal_'):
                        avg_sim = (sim_matrix.sum() - B) / (B * (B - 1) + 1e-8)
                        epoch_causal_sim += avg_sim.item() * B
                    else:
                        epoch_causal_sim += all_pairs_sim.mean().item() * B
                    total_graphs += B

                z_graph1_norm = F.normalize(z_graph1, dim=1)
                z_graph2_norm = F.normalize(z_graph2, dim=1)
                logits = (1.0 / 0.07) * z_graph1_norm @ z_graph2_norm.T
                logits.masked_fill(fn_mask, -1e9)
                loss = F.cross_entropy(logits, torch.arange(B, device=device))

            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        epoch_loss = loss_all / len(dataloader_aug)
        precision = epoch_tp / (epoch_tp + epoch_fp + 1e-8) if phase == 'FN_CL' else 0
        recall = epoch_tp / (epoch_tp + epoch_fn + 1e-8) if phase == 'FN_CL' else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if phase == 'FN_CL' else 0
        
        avg_predicted_fn_ratio = epoch_predicted_fn / (epoch_candidate_pairs + 1e-8) if phase == 'FN_CL' else 0.0
        avg_causal_sim = epoch_causal_sim / (total_graphs + 1e-8) if phase == 'FN_CL' else 0.0
        
        if phase == 'FN_CL':
            print(f'Epoch: {epoch:03d} [{phase}], Loss: {epoch_loss:.4f}, FN Prec: {precision:.4f}, FN Rec: {recall:.4f}, FN F1: {f1:.4f}, Pred FN: {epoch_predicted_fn} (Ratio: {avg_predicted_fn_ratio:.4f}), Avg Causal Sim: {avg_causal_sim:.4f}')
            # 🚀 Save FN metrics to TensorBoard
            writer.add_scalar('FN_Metrics/Precision', precision, epoch)
            writer.add_scalar('FN_Metrics/Recall', recall, epoch)
            writer.add_scalar('FN_Metrics/F1', f1, epoch)
            writer.add_scalar('FN_Metrics/Predicted_FN_Count', epoch_predicted_fn, epoch)
            writer.add_scalar('FN_Metrics/Predicted_FN_Ratio', avg_predicted_fn_ratio, epoch)
            writer.add_scalar('FN_Metrics/Causal_Similarity_Mean', avg_causal_sim, epoch)
        else:
            print(f'Epoch: {epoch:03d} [{phase}], Loss: {epoch_loss:.4f}')

        if epoch % args.log_interval == 0:
            current_eval_mode = 'groupvit' if phase == 'DISTILL' else args.eval_mode
            val_acc, test_acc = model_eval(model, dataloader_eval, device, current_eval_mode, args.rw_length, max_nodes=args.max_nodes)
            print(f'[{current_eval_mode}] Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            writer.add_scalar(f'Acc_{current_eval_mode}/Val', val_acc, epoch)
            writer.add_scalar(f'Acc_{current_eval_mode}/Test', test_acc, epoch)

        # [Strict Unsupervised] Just use the last epoch's weights, no val_acc selection
        if phase == 'DISTILL' and rel_epoch == args.distill_epochs_per_cycle - 1:
            print(f"--- End of DISTILL phase. Saving last epoch weights ---")
            torch.save(base_model.state_dict(), 'last_distill.pth')
            
        if phase == 'FN_CL' and rel_epoch == cycle_len - 1:
            print(f"--- End of FN_CL phase. Saving last epoch weights ---")
            torch.save(base_model.state_dict(), 'last_fn_cl.pth')

    writer.close()

if __name__ == '__main__':
    train()
