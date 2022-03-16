feat_init = ""
view_init = ""
flow_init = ""
emb2D_init = ""
vis_init = ""
occ_init = ""
ego_init = ""
emb_dim = 8
# occ_cheap = False
feat_dim = 32
feat_do_vae = False

view_depth = 32
view_pred_rgb = True
view_use_halftanh = True
view_pred_embs = False

occ_do_cheap = False
# this is the previous winner net, from which i was able to train a great flownet in 500i
feat_init = "04_m128x32x128_p64x192_1e-3_F32fr_Oc_c1_s1_V_d32_c1_E_s.1_a1_b.1_i1_j.1_caus2i6c1o0t_b13"
occ_init = "04_m128x32x128_p64x192_1e-3_F32fr_Oc_c1_s1_V_d32_c1_E_s.1_a1_b.1_i1_j.1_caus2i6c1o0t_b13"
view_init = "04_m128x32x128_p64x192_1e-3_F32fr_Oc_c1_s1_V_d32_c1_E_s.1_a1_b.1_i1_j.1_caus2i6c1o0t_b13"
vis_init = ""
flow_init = "04_m128x32x128_p64x192_1e-3_F32fr_Oc_c1_s1_V_d32_c1_E_s.1_a1_b.1_i1_j.1_caus2i6c1o0t_b13"
tow_init = ""

