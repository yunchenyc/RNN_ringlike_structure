import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import settings.task_setting as ts
import matplotlib_venn


color_map_sp, color_map_md = ts.color_map_sp, ts.color_map_md

def f1_inputs_outputs_overview(input_to_plot, output_to_plot, input_type, 
                                  save_path=None):
    
    # input_to_plot.shape = (n_trial, n_time, n_input)
    n_input = input_to_plot.shape[2]
    n_col = 5
    n_row = int(np.ceil(n_input / 2)) + 1

    plt.figure(figsize=(12, 12), dpi=300)
    for irow in range(n_row):
        for isub, idx in enumerate(range(n_col)):
            if irow < n_row - 1:
                plt.subplot(n_row, n_col, isub + 1 + irow * n_col)
                if ((irow + 1) * 2) <= n_input:
                    plt.plot(input_to_plot[idx, :, irow * 2:(irow + 1) * 2])
                else:
                    plt.plot(input_to_plot[idx, :, irow * 2:])
                plt.ylim(-2, 2)
                plt.axis('off')
            else:
                plt.subplot(n_row, n_col, isub + 1 + irow * n_col)
                plt.plot(output_to_plot[idx, :, 0])
                plt.plot(output_to_plot[idx, :, 1])
                plt.ylim(-3, 3)
                plt.axis('off')

    plt.suptitle(input_type)
    plt.subplots_adjust(top=0.9, wspace=0.3)
    if save_path:
        plt.savefig(save_path + 'f1_inputs_outputs_overview.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f2_inputs(input_to_plot, input_type, cd, save_path=None):
    plt.figure(figsize=(2, 1.5), dpi=300)
    plt.subplot(311)
    plt.plot(input_to_plot[0, :, :2])
    plt.vlines(x=cd.time_dict['go'][0], ymin=-1, ymax=1, 
               color='k', ls='--')
    plt.vlines(x=cd.time_dict['mo'][0], ymin=-1, ymax=1, 
               color='g', ls='--')
    plt.axis('off')
    plt.ylim(-0.2, 0.2)
    plt.subplot(312)
    # plt.plot(input_to_plot[0, :, 2:4])
    plt.plot(input_to_plot[0, :, -3:-1])
    plt.vlines(x=cd.time_dict['go'][0], ymin=-1, ymax=1, 
               color='k', ls='--')
    plt.vlines(x=cd.time_dict['mo'][0], ymin=-1, ymax=1, 
               color='g', ls='--')
    plt.axis('off')
    plt.ylim(-1.05, 1.05)
    plt.subplot(313)
    plt.plot(input_to_plot[0, :, -1], color='k')
    plt.vlines(x=cd.time_dict['go'][0], 
               ymin=-0.5, ymax=1.5, color='k', ls='--')
    plt.vlines(x=cd.time_dict['mo'][0], 
               ymin=-0.5, ymax=1.5, color='g', ls='--')
    plt.axis('off')
    # plt.suptitle('mPG')
    plt.suptitle(input_type)
    plt.subplots_adjust(top=0.8)
    if save_path:
        plt.savefig(save_path + 'f2_inputs.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f3_outputs(output_to_plot, cd, save_path=None):
    plt.figure(figsize=(2, 1.5), dpi=300)
    plt.subplot(111)
    plt.plot(output_to_plot[0, :, :])
    plt.vlines(x=cd.time_dict['go'][0], ymin=-3, ymax=1, color='k', ls='--')
    plt.vlines(x=cd.time_dict['mo'][0], ymin=-3, ymax=1, color='g', ls='--')
    plt.axis('off')
    plt.subplots_adjust(top=0.8)
    if save_path:
        plt.savefig(save_path + 'f3_outputs.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f4_output_compared_with_target(p_to_plot, y_to_plot, save_path=None):
    
    idxlist = range(p_to_plot.shape[0])
    if p_to_plot.shape[0]>18:
        idxlist = np.random.choice(idxlist, 18)
        
    plt.figure(figsize=(16, 12), dpi=300)
    for isub, idx in enumerate(idxlist):
        plt.subplot(3, 6, isub + 1)
        plt.plot(np.arange(y_to_plot.shape[1]), y_to_plot[isub, :, :], 
                 ls='--', label='Target')
        plt.plot(np.arange(p_to_plot.shape[1]), p_to_plot[isub, :, :], 
                 label='Model')

        plt.title('Trial %d' % idx)
        plt.ylim(1.2 * y_to_plot.min() - 0.3, 1.2 * y_to_plot.max() + 0.3)
        plt.xlim(-5, p_to_plot.shape[1] + 5)

    plt.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.1,
                        wspace=0.6, hspace=0.8)
    if save_path:
        plt.savefig(save_path + 'f4_output_compared_with_target.png', dpi=300)
        plt.show()
    else:
        plt.show()



def f5_hand_trajectory(target_pos, hand_pos, time_total_list, trial_info, 
                       save_path=None):
    
    plt.figure(figsize=(8, 2), dpi=300)
    for isub in range(target_pos.shape[0]):
        plt.subplot(1, 5, isub + 1)

        scatter_list = []
        cmap = plt.get_cmap('Greens', time_total_list[isub])
        cmap2 = plt.get_cmap('Blues', time_total_list[isub])
        for itime in range(time_total_list[isub]):
            if (itime + 1) % 10 == 0:
                plt.scatter(target_pos[isub, itime, 0], 
                            target_pos[isub, itime, 1],
                            facecolors=cmap(itime), edgecolors='none')

                plt.scatter(hand_pos[isub, itime, 0], 
                            hand_pos[isub, itime, 1],
                            facecolors=cmap2(itime), edgecolors='none', s=8)

            if itime == max(time_total_list) - 1:
                scatter_a = plt.scatter(target_pos[isub, itime, 0], 
                                        target_pos[isub, itime, 1],
                                        facecolors=cmap(itime), 
                                        edgecolors='none')
                scatter_list.append(scatter_a)

                scatter_i = plt.scatter(hand_pos[isub, itime, 0], 
                                        hand_pos[isub, itime, 1],
                                        facecolors=cmap2(itime), 
                                        edgecolors='none', s=8)
                scatter_list.append(scatter_i)

        plt.scatter(target_pos[isub, time_total_list[isub] - 1, 0], 
                    target_pos[isub, time_total_list[isub] - 1, 1],
                    s=50, c='g')
        plt.scatter(hand_pos[isub, time_total_list[isub] - 1, 0], 
                    hand_pos[isub, time_total_list[isub] - 1, 1],
                    s=50, c='b')
        plt.plot(np.cos(np.linspace(0, np.pi * 2, 100)) * ts.R_CIRCLE,
                 np.sin(np.linspace(0, np.pi * 2, 100)) * ts.R_CIRCLE,
                 ls='--', c='k')

        # plt.legend(scatter_list, ['target pos'],
        #            bbox_to_anchor=(1.1,  1.2), loc="upper left", ncol=2)
        plt.axis('off')
        plt.xlim(-0.17, 0.17)
        plt.ylim(-0.17, 0.17)
        ax = plt.gca()
        ax.set_aspect(1)

        plt.title(trial_info[isub])
        plt.subplots_adjust(hspace=0.8, right=0.6)
   
    if save_path:
        plt.savefig(save_path + 'f5_hand_trajectory.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f6_groupby_distance_error(error_df, color1, color2, xtext1, xtext2,
                              save_path=None):
    plt.figure(figsize=(10, 3), dpi=300)
    plt.subplot(121)
    sns.boxplot(data=error_df, x='SPIdx', y='distance_error', palette=color1)
    # plt.ylim(0, 0.05)
    # plt.subplots_adjust(left=0.05, right=0.95)
    # plt.savefig(PATH_SAVE + 'error_boxplot.png')
    plt.xlabel('Target speeds')
    plt.xticks(range(len(xtext1)), xtext1)
    
    plt.subplot(122)
    # sns.boxplot(data=error_df.query('SpeedIdx==2'), x='Type', 
    # y='direction_error', hue='InitType')
    sns.boxplot(data=error_df, x='MDIdx', y='distance_error', palette=color2)
    plt.xlabel('Movement direction groups')
    plt.xticks(range(len(xtext2)), xtext2)
    plt.subplots_adjust(wspace=0.5)
    if save_path:
        plt.savefig(save_path + 'f6_groupby_distance_error.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f7_condition_distribution(SPNum, MDNum, save_path=None):
    plt.figure(figsize=(7, 3), dpi=300)
    plt.subplot(121)
    plt.pie(SPNum, colors=[color_map_sp(i) for i in range(len(SPNum))],
            autopct='%1.1f%%', textprops={'fontsize': 10, 'color': 'w'})
    plt.subplot(122)
    plt.pie(MDNum, colors=[color_map_md(i) for i in range(len(MDNum))],
            autopct='%1.1f%%', textprops={'fontsize': 9, 'color': 'w'})
    plt.subplots_adjust(wspace=0.4)
    if save_path:
        plt.savefig(save_path + 'f7_condition_distribution.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f8_modulation_venn(to_modul_np, go_modul_np, mo_modul_np, save_path=None):
    
    plt.figure(figsize=(10, 4), dpi=300)
    plt.subplot(131)
    matplotlib_venn.venn3(
        subsets=[set(np.argwhere(to_modul_np[:, 0] == 1).flatten()),
                  set(np.argwhere(to_modul_np[:, 1] == 1).flatten()),
                  set(np.argwhere(to_modul_np[:, 2] == 1).flatten())],
        set_labels=('', '', ''),
        set_colors=("#adcdea", "#ff7f7f", "#ffdf7f"),
        alpha=0.4,
        normalize_to=1.0,
    )
    plt.title('TO')
    
    plt.subplot(132)
    matplotlib_venn.venn3(
        subsets=[set(np.argwhere(go_modul_np[:, 0] == 1).flatten()),
                  set(np.argwhere(go_modul_np[:, 1] == 1).flatten()),
                  set(np.argwhere(go_modul_np[:, 2] == 1).flatten())],
        set_labels=('', '', ''),
        set_colors=("#adcdea", "#ff7f7f", "#ffdf7f"),
        alpha=0.4,
        normalize_to=1.0,
    )
    plt.title('GO')
    
    plt.subplot(133)
    matplotlib_venn.venn3(
        subsets=[set(np.argwhere(mo_modul_np[:, 0] == 1).flatten()),
                  set(np.argwhere(mo_modul_np[:, 1] == 1).flatten()),
                  set(np.argwhere(mo_modul_np[:, 2] == 1).flatten())],
        set_labels=('', '', ''),
        set_colors=("#adcdea", "#ff7f7f", "#ffdf7f"),
        alpha=0.4,
        normalize_to=1.0,
    )
    plt.title('MO')
    
    if save_path:
        plt.savefig(save_path + 'f8_modulation_venn.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f9_neuronal_modulation_radar(r_np, nodeid, nSP, nMD, save_path=None):
    plt.figure(figsize=(3, 3), dpi=300)
    ax = plt.subplot(111, projection='polar')
    
    pd_np = np.zeros((nSP,))
    mdd = np.arange(0, np.pi * 2, np.pi / 4)
    for isp in range(nSP):
        cossum = np.sum(np.cos(mdd) * r_np[isp * nMD:(isp + 1) * nMD])
        sinsum = np.sum(np.sin(mdd) * r_np[isp * nMD:(isp + 1) * nMD])
        pd_np[isp] = np.arctan2(sinsum, cossum)

    # plot radar tuning curves
    ax.plot(np.linspace(0, np.pi * 2, 100), np.ones((100,)), 'k-', lw=3)
    ax.scatter(0, 0, s=40, c='k')
    plotdd = np.arange(0, np.pi * 2 + np.pi / 4, np.pi / 4)
    for isp in range(nSP):
        ax.plot([pd_np[isp], pd_np[isp]], [1.1, 1.3],
                color=color_map_sp(isp), lw=3)
        vlen = r_np[isp * nMD:(isp + 1) * nMD].tolist()
        vlen.append(r_np[isp * nMD])
        vlen = np.array(vlen) / r_np.max()
        ax.plot(plotdd, vlen, color=color_map_sp(isp), lw=3)
    # ax.set_xlim(-1.1, 1.1)
    # ax.set_ylim(-1.1, 1.1)
    ax.set_aspect(1)
    ax.axis('off')

    plt.title('Node %d' % nodeid)
    
    if save_path:
        plt.savefig(save_path + 'f9_neuronal_modulation_radar_%d.png' % nodeid, 
                    dpi=300)
        plt.show()
    else:
        plt.show()


def f10_neuronal_conditional_rates(rate_plot_single, nodeid, SPIdx, MDIdx, 
                                   save_path=None):
    plt.figure(figsize=(8, 8), dpi=300)
    
    nSP = len(np.unique(SPIdx))
    nMD = len(np.unique(MDIdx))
    sublist = [8, 3, 2, 5, 9, 14, 15, 12]
    for isp in range(nSP):
        for imd in range(nMD):
            plt.subplot(4, 4, sublist[imd])
                
            temp = rate_plot_single[(SPIdx == isp) & (MDIdx == imd), :]
            
            plt.vlines(0, ymin=0, ymax=1, color='k', lw=0.5)
            plt.plot(np.arange(-300, 150), np.mean(temp, axis=0), 
                     color=color_map_sp(isp))
            plt.fill_between(
                np.arange(-300, 150),
                np.mean(temp, axis=0) - 2 * np.std(temp, axis=0)/temp.shape[0],
                np.mean(temp, axis=0) + 2 * np.std(temp, axis=0)/temp.shape[0],
                facecolor=color_map_sp(isp), alpha=0.2)
            plt.ylim(0, 1.2 * rate_plot_single.max())
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            ax = plt.gca()
            # ax.set_aspect(1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.title('Node %d at %d' % (nodeid, imd))
    if save_path:
        plt.savefig(save_path 
                    + 'f10_neuronal_conditional_rates_%d.png' % nodeid, 
                    dpi=300)
        plt.show()
    else:
        plt.show()


def f11_neural_states_3d(pcs, evr, group_idx, group_color, group, 
                         save_path=None):
    plt.figure(figsize=(5, 5), dpi=300)
    ax = plt.subplot(111, projection='3d')
    ax.scatter3D(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=group_color(group_idx))
    # ax.scatter3D(xx0, yy0, zz0, c='r')
    ax.set_xlabel('PC1 (%.1f' % (evr[0] * 100) + '%)')
    ax.set_ylabel('PC2 (%.1f' % (evr[1] * 100) + '%)')
    ax.set_zlabel('PC3 (%.1f' % (evr[2] * 100) + '%)')
    ax.view_init(elev=60, azim=-60)
    plt.title(group)
    if save_path:
        plt.savefig(save_path + 'f11_neural_states_3d_%s.png' % group, dpi=300)
        plt.show()
    else:
        plt.show()
  
    
def f12_neural_states_2d(pcs, evr, group_idx, group_color, group, 
                         save_path=None):
    plt.figure(figsize=(10, 3), dpi=300)
    plt.subplot(131)
    plt.scatter(pcs[:, 0], pcs[:, 1],
                facecolors=group_color(group_idx))
    plt.xlabel('PC1 (%.1f' % (evr[0] * 100) + '%)')
    plt.ylabel('PC2 (%.1f' % (evr[1] * 100) + '%)')
    plt.xlim(-pcs.max() * 1.2, pcs.max() * 1.2)
    plt.ylim(-pcs.max() * 1.2, pcs.max() * 1.2)
    ax = plt.gca()
    ax.set_aspect(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    plt.scatter(pcs[:, 0], pcs[:, 2],
                facecolors=group_color(group_idx))
    plt.xlabel('PC1 (%.1f' % (evr[0] * 100) + '%)')
    plt.ylabel('PC3 (%.1f' % (evr[2] * 100) + '%)')
    plt.xlim(-pcs.max() * 1.2, pcs.max() * 1.2)
    plt.ylim(-pcs.max() * 1.2, pcs.max() * 1.2)
    ax = plt.gca()
    ax.set_aspect(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    plt.scatter(pcs[:, 1], pcs[:, 2],
                facecolors=group_color(group_idx))
    plt.xlabel('PC2 (%.1f' % (evr[1] * 100) + '%)')
    plt.ylabel('PC3 (%.1f' % (evr[2] * 100) + '%)')
    plt.xlim(-pcs.max() * 1.2, pcs.max() * 1.2)
    plt.ylim(-pcs.max() * 1.2, pcs.max() * 1.2)
    ax = plt.gca()
    ax.set_aspect(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    # plt.legend()
    plt.subplots_adjust(wspace=0.4)
    plt.suptitle(group)
    
    if save_path:
        plt.savefig(save_path + 'f12_neural_states_2d_%s.png' % group, dpi=300)
        plt.show()
    else:
        plt.show()


def f13_states_fitted_plane(points, planes, group, save_path=None):
    
    plt.figure(figsize=(5, 4))
    ax = plt.subplot(111, projection='3d')
    
    if isinstance(points, list) and isinstance(planes, list):
        for (pp, pl) in zip(points, planes):
            x, y, z = pp
            x_plane, y_plane, z_plane = pl
            ax.scatter3D(x, y, z)
            ax.plot_wireframe(x_plane, y_plane, z_plane, rstride=10, cstride=10)
            # ax.plot_surface(x_plane, y_plane, z_plane, rstride=5, cstride=5, 
            #                 cmap='rainbow', alpha=0.5)
        
    else:
        x, y, z = points
        x_plane, y_plane, z_plane = planes
        ax.scatter3D(x, y, z)
        ax.plot_wireframe(x_plane, y_plane, z_plane, rstride=10, cstride=10)
        # ax.plot_surface(x_plane, y_plane, z_plane, rstride=5, cstride=5, 
        #                 cmap='rainbow', alpha=0.5)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=60, azim=-60)
    plt.title(group)
    
    if save_path:
        plt.savefig(save_path + 'f13_states_fitted_plane_%s.png' % group, 
                    dpi=300)
        plt.show()
    else:
        plt.show()


def f14_states_fitted_ellipses(pcs_points, evr, el_points, group_idx, 
                                       group_color, group, save_path=None):
    
    isp = 0

    plt.figure(figsize=(20, 6), dpi=300)
    ax1 = plt.subplot(141, projection='3d')
    ax2 = plt.subplot(142)
    ax3 = plt.subplot(143)
    ax4 = plt.subplot(144)
    
    if isinstance(pcs_points, tuple) and isinstance(el_points, tuple):
        pcs_points = [pcs_points]
        el_points = [el_points]
    
    if isinstance(pcs_points, list) and isinstance(el_points, list):
        assert (len(pcs_points)==len(el_points) 
                and len(pcs_points)==len(np.unique(group_idx)))
        
        for isp, (pp, elp) in enumerate(zip(pcs_points, el_points)):
            ax1.scatter3D(pp[0], pp[1], pp[2], 
                         c=group_color(group_idx[group_idx==isp]))
            ax1.plot3D(elp[0], elp[1], elp[2], 
                      c=group_color(isp))
            
            ax2.scatter(pp[0], pp[1], c=group_color(group_idx[group_idx==isp]))
            ax2.scatter(elp[0], elp[1], 
                       c=group_color((np.ones_like(elp[0])*isp).astype(int)), 
                       alpha=0.2)
            
            ax3.scatter(pp[0], pp[2], c=group_color(group_idx[group_idx==isp]))
            ax3.scatter(elp[0], elp[2], 
                       c=group_color((np.ones_like(elp[0])*isp).astype(int)),
                       alpha=0.2)
            
            ax4.scatter(pp[1], pp[2], c=group_color(group_idx[group_idx==isp]))
            ax4.scatter(elp[1], elp[2], 
                       c=group_color((np.ones_like(elp[0])*isp).astype(int)), 
                       alpha=0.2)
       
    pcs_max = max([(np.asarray(i)).max() for i in pcs_points])
    # ax.scatter3D(xx0, yy0, zz0, c='r')
    ax1.set_xlabel('PC1 (%.1f' % (evr[0] * 100) + '%)')
    ax1.set_ylabel('PC2 (%.1f' % (evr[1] * 100) + '%)')
    ax1.set_zlabel('PC3 (%.1f' % (evr[2] * 100) + '%)')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=60, azim=-60)
    
    ax2.set_xlabel('PC1 (%.1f' % (evr[0] * 100) + '%)')
    ax2.set_ylabel('PC2 (%.1f' % (evr[1] * 100) + '%)')
    ax2.set_xlim(-pcs_max*1.2, pcs_max*1.2)
    ax2.set_ylim(-pcs_max*1.2, pcs_max*1.2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect(1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax3.set_xlabel('PC1 (%.1f' % (evr[0] * 100) + '%)')
    ax3.set_ylabel('PC3 (%.1f' % (evr[2] * 100) + '%)')
    ax3.set_xlim(-pcs_max*1.2, pcs_max*1.2)
    ax3.set_ylim(-pcs_max*1.2, pcs_max*1.2)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect(1)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
   
    ax4.set_xlabel('PC2 (%.1f' % (evr[1] * 100) + '%)')
    ax4.set_ylabel('PC3 (%.1f' % (evr[2] * 100) + '%)')
    ax4.set_xlim(-pcs_max*1.2, pcs_max*1.2)
    ax4.set_ylim(-pcs_max*1.2, pcs_max*1.2)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_aspect(1)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    
    plt.suptitle(group)
    
    if save_path:
        plt.savefig(save_path + 'f14_states_fitted_ellipses_%s.png' % group, 
                    dpi=300)
        plt.show()
    else:
        plt.show()


def f15_angle_between_planes(x, y, slope, intercept, lf, c, save_path=None):
    plt.figure(figsize=(4, 3), dpi=300)
    # plt.bar(SPList/np.pi*180, theta_pp_np, width=10, 
    # color=[color_map_sp(i) for i in range(nSP)])
    plt.scatter(x, y, color=c)
    plt.plot(np.linspace(-360, 360, 100),
             lf(np.linspace(-360, 360, 100), slope, intercept), 'k--')
    plt.xticks([-240, -120, 0, 120, 240])
    plt.title('y = %.2fx + %.2f' % (slope, intercept))
    ax = plt.gca()
    # ax.set_aspect(6)
    # plt.ylim(0, 90)
    # ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if save_path:
        plt.savefig(save_path + 'f15_angle_between_planes.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f16_angle_elevation_azimuth(x, theta_z_np, theta_xy_np, co_idx, c, 
                                save_path=None):
    plt.figure(figsize=(4, 3), dpi=300)
    plt.bar(x, theta_z_np - theta_z_np[co_idx] + 90,
            width=[1.5, 1.5, 1.5, 1.5, 1.5], color=c)
    plt.bar(x + 12, 
            np.mod(theta_xy_np - theta_xy_np[co_idx] + 90, 360),
            width=[1.5, 1.5, 1.5, 1.5, 1.5], color=c)
    
    plt.yticks([0., 90, 180, 270])
    plt.xticks([0, 12], labels=['A', 'B'])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_aspect(1)
    # plt.axis('off')
    # plt.title('z angle of norm vectors')
    plt.subplots_adjust(wspace=0.4)
    
    if save_path:
        plt.savefig(save_path + 'f16_angle_elevation_azimuth.png', dpi=300)
        plt.show()
    else:
        plt.show()


def f17_connectivity_between_modulations(w_hh, dict_mol, mol, save_path=None):
    plt.figure(figsize=(10, 2), dpi=300)
    for isub, iperiod in enumerate(dict_mol.keys()):
        connec_mol = np.zeros((3, 3))
        for ii, itype in enumerate(mol):
            for jj, jtype in enumerate(mol):
                temp = []
                for ia in dict_mol[iperiod][itype]:
                    temp.append(np.mean(abs(w_hh[dict_mol[iperiod][jtype], ia])))
                connec_mol[ii, jj] = np.mean(np.asarray(temp))
        plt.subplot(1, 3, isub + 1)
        sns.heatmap(connec_mol)
        plt.xticks([0.5, 1.5, 2.5], mol)
        plt.yticks([0.5, 1.5, 2.5], mol)
        plt.title(iperiod)
    plt.subplots_adjust(wspace=0.4)
    
    if save_path:
        plt.savefig(save_path + 'f17_connectivity_between_modulations.png', 
                    dpi=300)
        plt.show()
    else:
        plt.show()


def f18_neural_states_perturbed(pcs, evr, group_idx, group_color, group, 
                                save_path=None):
    
    # same as f11 only different filename
    
    plt.figure(figsize=(5, 5), dpi=300)
    ax = plt.subplot(111, projection='3d')
    ax.scatter3D(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=group_color(group_idx))
    # ax.scatter3D(xx0, yy0, zz0, c='r')
    ax.set_xlabel('PC1 (%.1f' % (evr[0] * 100) + '%)')
    ax.set_ylabel('PC2 (%.1f' % (evr[1] * 100) + '%)')
    ax.set_zlabel('PC3 (%.1f' % (evr[2] * 100) + '%)')
    ax.view_init(elev=60, azim=-60)
    plt.title(group)
    if save_path:
        plt.savefig(save_path + 'f18_neural_states_perturbed_%s.png' % group, dpi=300)
        plt.show()
    else:
        plt.show()
        

def f19_hand_trajectory(target_pos, hand_pos, time_total_list, trial_info, 
                       note, save_path=None):
    
    plt.figure(figsize=(8, 2), dpi=300)
    for isub in range(target_pos.shape[0]):
        plt.subplot(1, 5, isub + 1)

        scatter_list = []
        cmap = plt.get_cmap('Greens', time_total_list[isub])
        cmap2 = plt.get_cmap('Blues', time_total_list[isub])
        for itime in range(time_total_list[isub]):
            if (itime + 1) % 10 == 0:
                plt.scatter(target_pos[isub, itime, 0], 
                            target_pos[isub, itime, 1],
                            facecolors=cmap(itime), edgecolors='none')

                plt.scatter(hand_pos[isub, itime, 0], 
                            hand_pos[isub, itime, 1],
                            facecolors=cmap2(itime), edgecolors='none', s=8)

            if itime == max(time_total_list) - 1:
                scatter_a = plt.scatter(target_pos[isub, itime, 0], 
                                        target_pos[isub, itime, 1],
                                        facecolors=cmap(itime), 
                                        edgecolors='none')
                scatter_list.append(scatter_a)

                scatter_i = plt.scatter(hand_pos[isub, itime, 0], 
                                        hand_pos[isub, itime, 1],
                                        facecolors=cmap2(itime), 
                                        edgecolors='none', s=8)
                scatter_list.append(scatter_i)

        plt.scatter(target_pos[isub, time_total_list[isub] - 1, 0], 
                    target_pos[isub, time_total_list[isub] - 1, 1],
                    s=50, c='g')
        plt.scatter(hand_pos[isub, time_total_list[isub] - 1, 0], 
                    hand_pos[isub, time_total_list[isub] - 1, 1],
                    s=50, c='b')
        plt.plot(np.cos(np.linspace(0, np.pi * 2, 100)) * ts.R_CIRCLE,
                 np.sin(np.linspace(0, np.pi * 2, 100)) * ts.R_CIRCLE,
                 ls='--', c='k')

        # plt.legend(scatter_list, ['target pos'],
        #            bbox_to_anchor=(1.1,  1.2), loc="upper left", ncol=2)
        plt.axis('off')
        plt.xlim(-0.17, 0.17)
        plt.ylim(-0.17, 0.17)
        ax = plt.gca()
        ax.set_aspect(1)

        plt.title(trial_info[isub])
        plt.subplots_adjust(hspace=0.8, right=0.6)
    
    plt.suptitle(note)
    
    if save_path:
        plt.savefig(save_path + 'f19_hand_trajectory_blocking_%s.png' % note, 
                    dpi=300)
        plt.show()
    else:
        plt.show()
        