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
    Peforms random walks per graph to avoid OOM on large batches.
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
    parser.add_argument('--fn_mode', type=str, default='causal_sim', choices=['sim', 'rbo', 'causal_sim', 'causal_rbo'])
    parser.add_argument('--causal_k', type=int, default=2, help='Top K components to form causal subgraph representation')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--rbo_threshold', type=float, default=0.6)
    parser.add_argument('--rbo_p', type=float, default=0.98)
    parser.add_argument('--tensorboard_dir', type=str, default='runs/graphvit_cotrain')
    parser.add_argument('--attn_mode', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--max_nodes', type=int, default=1024, help='Max nodes per graph to prevent OOM')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU DataParallel training')
    
    # 🚀 Co-training specific arguments
    parser.add_argument('--warmup_epochs', type=int, default=40, help='Epochs to transition alpha from 0.0 to 1.0 smoothly')
    parser.add_argument('--distill', action='store_true', default=True, help='Enable GCL teacher-student distillation')
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--distill_temp', type=float, default=0.2)
    
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
    
    # Always use augmented data for training to ensure feature quality
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
    # 🚀 Unfreeze both GCL and GroupViT throughout co-training
    model.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(teacher_proj.parameters()), lr=args.lr)
    
    best_val_acc = 0.0

    print("==========================================================")
    print(f"Co-Training Strategy Activated: Warmup Epochs: {args.warmup_epochs}")
    print("==========================================================")

    for epoch in range(1, args.epochs + 1):
        # 🚀 1. Compute dynamic alpha weight smoothly
        if args.warmup_epochs > 0:
            alpha = min(1.0, epoch / float(args.warmup_epochs))
        else:
            alpha = 1.0

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
            
            z_nodes1, z_graph1, components1, attn_list1, rw_dense1 = model(data1, max_nodes=args.max_nodes)
            z_nodes2, z_graph2, components2, attn_list2, rw_dense2 = model(data2, max_nodes=args.max_nodes)
            
            B = z_nodes1.shape[0]
            z_nodes1_norm = F.normalize(z_nodes1, dim=1)
            z_nodes2_norm = F.normalize(z_nodes2, dim=1)

            # ----------------------------------------------------
            # 🚀 Loss A: Teacher Distillation Loss (Node level)
            # ----------------------------------------------------
            loss_self = F.cross_entropy((1.0 / 0.07) * z_nodes1_norm @ z_nodes2_norm.T, torch.arange(B, device=device))
            loss_teacher = loss_self
            
            if args.distill and teacher is not None:
                with torch.no_grad():
                    z_teacher, _ = teacher(data1.x, data1.edge_index, data1.batch)
                    z_teacher_proj = F.normalize(teacher_proj(z_teacher), dim=1)
                
                logits_to_teacher = (1.0 / 0.07) * z_nodes1_norm @ z_teacher_proj.T
                loss_to_teacher = F.cross_entropy(logits_to_teacher, torch.arange(B, device=device))
                
                sim_T = torch.softmax(z_teacher_proj @ z_teacher_proj.T / args.distill_temp, dim=-1)
                sim_S = torch.log_softmax(z_nodes1_norm @ z_nodes1_norm.T / args.distill_temp, dim=-1)
                loss_distill = F.kl_div(sim_S, sim_T, reduction='batchmean')
                
                loss_teacher = loss_teacher + loss_to_teacher + args.distill_weight * loss_distill

            # ----------------------------------------------------
            # 🚀 Loss B: Reweighted GCL Loss (Graph level)
            # ----------------------------------------------------
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

            # Evaluate against true labels for metrics reporting
            true_fn_mask = (data1.y.unsqueeze(0) == data1.y.unsqueeze(1))
            true_fn_mask.fill_diagonal_(False)
            epoch_tp += (fn_mask & true_fn_mask).sum().item()
            epoch_fp += (fn_mask & ~true_fn_mask).sum().item()
            epoch_fn += (~fn_mask & true_fn_mask).sum().item()

            # Track predicted FN stats for verification
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
            logits_gcl = (1.0 / 0.07) * z_graph1_norm @ z_graph2_norm.T
            logits_gcl.masked_fill(fn_mask, -1e9)
            loss_infonce_reweight = F.cross_entropy(logits_gcl, torch.arange(B, device=device))

            # ----------------------------------------------------
            # 🚀 Unified Loss Co-training Formula with Warmup weight
            # ----------------------------------------------------
            loss = (1.0 - alpha) * loss_teacher + alpha * loss_infonce_reweight

            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        epoch_loss = loss_all / len(dataloader_aug)
        precision = epoch_tp / (epoch_tp + epoch_fp + 1e-8)
        recall = epoch_tp / (epoch_tp + epoch_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        avg_predicted_fn_ratio = epoch_predicted_fn / (epoch_candidate_pairs + 1e-8)
        avg_causal_sim = epoch_causal_sim / (total_graphs + 1e-8)
        
        print(f'Epoch: {epoch:03d}, alpha: {alpha:.3f}, Loss: {epoch_loss:.4f}, FN Prec: {precision:.4f}, FN Rec: {recall:.4f}, FN F1: {f1:.4f}, Pred FN: {epoch_predicted_fn} (Ratio: {avg_predicted_fn_ratio:.4f}), Avg Causal PeSim: {avg_causal_sim:.4f}')
        
        # Save logs to TensorBoard
        writer.add_scalar('Loss/total_loss', epoch_loss, epoch)
        writer.add_scalar('Loss/alpha', alpha, epoch)
        writer.add_scalar('FN_Metrics/Precision', precision, epoch)
        writer.add_scalar('FN_Metrics/Recall', recall, epoch)
        writer.add_scalar('FN_Metrics/F1', f1, epoch)
        writer.add_scalar('FN_Metrics/Predicted_FN_Count', epoch_predicted_fn, epoch)
        writer.add_scalar('FN_Metrics/Predicted_FN_Ratio', avg_predicted_fn_ratio, epoch)
        writer.add_scalar('FN_Metrics/Causal_Similarity_Mean', avg_causal_sim, epoch)

        if epoch % args.log_interval == 0:
            val_acc, test_acc = model_eval(model, dataloader_eval, device, args.eval_mode, args.rw_length, max_nodes=args.max_nodes)
            print(f'[{args.eval_mode}] Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            writer.add_scalar(f'Acc_{args.eval_mode}/Val', val_acc, epoch)
            writer.add_scalar(f'Acc_{args.eval_mode}/Test', test_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"--- New Best Val Acc: {best_val_acc:.4f}. Saving weights ---")
                torch.save(base_model.state_dict(), 'best_model_cotrain.pth')

    torch.save(base_model.state_dict(), 'last_model_cotrain.pth')
    writer.close()

if __name__ == '__main__':
    train()
