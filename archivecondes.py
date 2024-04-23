    
"""
iBot loss intermediate cls token clustering loss
"""

# class iBOTLoss_double(nn.Module):
#     def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
#                  teacher_temp, warmup_teacher_temp2, teacher_temp2, 
#                  warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
#                  center_momentum=0.9, center_momentum2=0.9,
#                  lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
#         super().__init__()
#         self.student_temp = student_temp
#         self.center_momentum = center_momentum
#         self.center_momentum2 = center_momentum2
#         self.ngcrops = ngcrops
#         self.nlcrops = nlcrops
#         self.ncrops = ngcrops + nlcrops
#         self.register_buffer("center", torch.zeros(3, out_dim))
#         self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2

#         # we apply a warm up for the teacher temperature because
#         # a too high temperature makes the training instable at the beginning
#         self.teacher_temp_schedule = np.concatenate((
#             np.linspace(warmup_teacher_temp,
#                         teacher_temp, warmup_teacher_temp_epochs),
#             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
#         ))
#         self.teacher_temp2_schedule = np.concatenate((
#             np.linspace(warmup_teacher_temp2,
#                         teacher_temp2, warmup_teacher_temp_epochs),
#             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
#         )) if mim_start_epoch == 0 else np.concatenate((
#             np.ones(mim_start_epoch) * warmup_teacher_temp2,
#             np.linspace(warmup_teacher_temp2,
#                         teacher_temp2, warmup_teacher_temp_epochs),
#             np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
#         ))

#     def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
#         """
#         Cross-entropy between softmax outputs of the teacher and student networks.
#         """
#         student_cls, student_patch = student_output
#         teacher_cls, teacher_patch = teacher_output
        
#         if student_local_cls is not None:
#             student_cls = torch.cat([student_cls, student_local_cls])

#         # [CLS] and patch for global patches
#         student_cls = student_cls / self.student_temp
#         student_cls_c = student_cls.chunk(self.ncrops)
#         student_patch = student_patch / self.student_temp
#         student_patch_c = student_patch.chunk(self.ngcrops)
        
#         # teacher centering and sharpening
#         temp = self.teacher_temp_schedule[epoch]
#         temp2 = self.teacher_temp2_schedule[epoch]
#         teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
#         teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
#         teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
#         teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

#         total_loss10, total_loss11, total_loss12 = 0, 0, 0

#         total_loss1, n_loss_terms1 = 0, 0
#         total_loss2, n_loss_terms2 = 0, 0
#         for q in range(len(teacher_cls_c)):
#             for v in range(len(student_cls_c)):
#                 if v == q: # if the crop is same!
#                     loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
#                     mask = student_mask[v].flatten(-2, -1)
#                     loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
#                     total_loss2 += loss2.mean()
#                     n_loss_terms2 += 1
#                 else: # predict one crop from another!
#                     loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)

#                     total_loss10 += loss1[:,0].mean()
#                     total_loss11 += loss1[:,1].mean()
#                     total_loss12 += loss1[:,2].mean()

#                     total_loss1 += loss1.mean()
#                     n_loss_terms1 += 1
            
#         total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
#         total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2

#         total_loss10 = total_loss10 / n_loss_terms1 * self.lambda1
#         total_loss11 = total_loss11 / n_loss_terms1 * self.lambda1
#         total_loss12 = total_loss12 / n_loss_terms1 * self.lambda1

#         total_loss = dict(c0=total_loss10, c1=total_loss11, c2=total_loss12, cls=total_loss1, patch=total_loss1, loss=total_loss1 + total_loss2)
#         self.update_center(teacher_cls, teacher_patch)                  
#         return total_loss

#     @torch.no_grad()
#     def update_center(self, teacher_cls, teacher_patch):
#         """
#         Update center used for teacher output.
#         """
#         cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
#         dist.all_reduce(cls_center)
#         cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
#         self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

#         patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
#         dist.all_reduce(patch_center)
#         patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
#         self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

"""
iBot head with 3 heads for each cls token
"""

# class iBOTHead(DINOHead):

#     def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', last_norm=None, 
#                  nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
#                  shared_head=False, **kwargs):
        
#         super(iBOTHead, self).__init__(*args,
#                                         norm=norm,
#                                         act=act,
#                                         last_norm=last_norm,
#                                         nlayers=nlayers,
#                                         hidden_dim=hidden_dim,
#                                         bottleneck_dim=bottleneck_dim,
#                                         norm_last_layer=norm_last_layer, 
#                                         **kwargs)

#         if not shared_head:
#             if bottleneck_dim > 0:
#                 self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
#                 self.last_layer2.weight_g.data.fill_(1)
#                 if norm_last_layer:
#                     self.last_layer2.weight_g.requires_grad = False
#             else:
#                 self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
#                 self.last_layer2 = None

#             self.last_norm2 = self._build_norm(last_norm, patch_out_dim, affine=False, **kwargs)
#         else:
#             if bottleneck_dim > 0:
#                 self.last_layer2 = self.last_layer
#             else:
#                 self.mlp2 = self.mlp[-1]
#                 self.last_layer2 = None

#             self.last_norm2 = self.last_norm


#     def forward(self, x):
#         if len(x.shape) == 2:
#             return super(iBOTHead, self).forward(x)

#         if self.last_layer is not None:
#             x1 = self.mlp_low1(x[:, 0:1])
#             x2 = self.mlp_low2(x[:, 1:2])
#             x3 = self.mlp(x[:, 2:])
#             x = torch.cat([x1, x2, x3], dim=1)

#             x = nn.functional.normalize(x, dim=-1, p=2)

#             x11 = self.last_layer_low1(x[:, 0:1])
#             x12 = self.last_layer_low2(x[:, 1:2])
#             x13 = self.last_layer(x[:, 2:3])
#             x1 = torch.cat([x11, x12, x13], dim=1)

#             x2 = self.last_layer2(x[:, 3:])
#         else:
#             x = self.mlp[:-1](x)
#             x1 = self.mlp[-1](x[:, 0:3])
#             x2 = self.mlp2(x[:, 3:])
        
#         if self.last_norm is not None:
#             x1 = self.last_norm(x1)
#             x2 = self.last_norm2(x2)
        
#         return x1, x2



"""
Vision transformer code to get class tokens from previous layers
"""

    # def forward(self, x, return_all_tokens=None, mask=None):
    #     # mim
    #     if self.masked_im_modeling:
    #         assert mask is not None
    #         x = self.prepare_tokens(x, mask=mask)
    #     else:
    #         x = self.prepare_tokens(x)

    #     class_idxs = [3, 7]
    #     cls_tokens = None
    #     for blk_idx, blk in enumerate(self.blocks):
    #         x = blk(x)
    #         if blk_idx in class_idxs:
    #             cls = self.norm(x[:, 0:1])
    #             cls_tokens = cls if cls_tokens is None else torch.cat([cls_tokens, cls], dim=1)
        
    #     x = self.norm(x)

    #     if self.fc_norm is not None:
    #         x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

    #     x = torch.cat([cls_tokens, x], dim=1)
        
    #     return_all_tokens = self.return_all_tokens if \
    #         return_all_tokens is None else return_all_tokens
    #     if return_all_tokens:
    #         return x
    #     return x[:,0:len(class_idxs)+1]

