import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn import svm

def linearize(corrmat):

    """Converts ROI x ROI x subjects representation into just the linearized upper triangle x subject

    
    Args:
        array (corrmat): 3d numpy array representing ROI x ROI x subject

    Returns:
        array: 2d linearized, upper triangle of the above corrmats representing upper triangles x subject

    """    
    if len(corrmat.shape) == 2 and corrmat.shape == (352,352):
        corrmat = np.expand_dims(corrmat,2)
    
    temp = []

    for ind in range(corrmat.shape[2]):

        single_subject = corrmat[:,:,ind]
        single_subject[np.where(single_subject == 0)] = 999
        
        upper_triangle_nonzero = np.triu(single_subject).flatten()[np.triu(single_subject).flatten().nonzero()]
        upper_triangle_nonzero[np.where(upper_triangle_nonzero == 999)] = 0

        temp.append(upper_triangle_nonzero)
    
    return np.vstack(temp)

def triangularizeweights(weights):
    """Converts weight or coefficient vector into the upper triangle on a 352x352 matrix for plotting purposes 

    Args:
        array (vector):  1x62128 flattened upper triangle

    Returns:
        array: ROIxROI corrmat  
    """   
    
    if weights.shape == (1, 62128):
        weights = weights.squeeze()
    
    result = np.zeros((352,352))
    ind = np.triu_indices(352)

    for i in range(len(weights)):
        result[ind[0][i],ind[1][i]] = weights[i]
    
    return result

def make_corrfig(z_trans_mat, weights = False):

    ## Get all the visuals info for this parcellation#
    # label_names = ['Auditory','CinguloOperc','CinguloParietal','Default','DorsalAttn','FrontoParietal','None',
    #             'RetrosplenialTemporal','Salience','SMhand','SMmouth','VentralAttn','Visual']
    names_abbrev = ['Auditory','CingOperc','CingPar','Default','DorsalAtt','FrontoPar','None',
                    'RetroTemp','Salience','SMhand','SMmouth','VentralAtt','Visual','Subcort']
    ## COLORS AND THINGS FOR CORR MATRIX IMAGE ##
    color_label_list = ['pink','purple','mediumorchid','red','lime','yellow','white',
                        'bisque','black','cyan','orange','teal','blue','brown']
    # Range for color bars
    range_list = ['0-24','24-64','64-69','69-110','110-142','142-166','166-213',
                '213-221','221-225','225-263','263-271','271-294','294-333','333-353']
    # CREATING numpy array of ranges #
    formatted_range_list = []
    for rl in range_list:
        rl = rl.split('-')
        rl =  map(int, rl)
        formatted_range_list.append(list(rl))
    labels = np.array(formatted_range_list)
    # Index list for lines
    line_list = [24,64,69,110,142,166,213,221,225,263,271,294,333]
#     fig, ax = plt.subplots(figsize=(20, 12))
    fig, ax = plt.subplots()
    if weights == True:
        im = ax.imshow(z_trans_mat, aspect='equal')
    else:
        cmap = cm.get_cmap('jet')
        low_thresh = -0.4
        high_thresh = 1.0
        im = ax.imshow(z_trans_mat, aspect='equal',cmap=cmap,vmin=low_thresh,vmax=high_thresh)
        ax.set_title('Z-Transformed Connectivity Matrix', fontsize=30)
        # TITLE AND COLORBAR ADJUSTMENTS #
        cbar = fig.colorbar(im, pad=0.0009)
        cbar.set_ticks([.8,.6,.4,.2,0,-0.2,-0.4,-0.6,-0.8])
        cbar.ax.set_ylabel('arctanh (z-transformed) values',rotation=270, labelpad=12, weight='bold')
        cbar.ax.tick_params(labelsize=5, pad=3)
        cbar.update_ticks()
        cbar.ax.yaxis.set_ticks_position('left')
    #DRAWING LINES
    for the_line in line_list:
        ax.axhline(y=the_line - .5, linewidth=1.5, color='white')
        ax.axvline(x=the_line - .5, linewidth=1.5, color='white')

    # CREATE AXES NEXT TO PLOT
    divider = make_axes_locatable(ax)
    axb = divider.append_axes("bottom", "10%", pad=0.02, sharex=ax)
    axl = divider.append_axes("left", "10%", pad=0.02, sharey=ax)
    axb.invert_yaxis()
    axl.invert_xaxis()
    axb.axis("off")
    axl.axis("off")
    # PLOT COLORED BARS TO THE AXES
    barkw = dict( color=color_label_list, linewidth=0.50, ec="k", clip_on=False, align='edge',)
    # bottom bar #
    axb.bar(labels[:,0]-.5,np.ones(len(labels)),
            width=np.diff(labels, axis=1).flatten(), **barkw)
    # side bar #
    axl.barh(labels[:,0]-.5,np.ones(len(labels)),
            height=np.diff(labels, axis=1).flatten(), **barkw)
    # SET MARGINS TO ZERO AGAIN
    ax.margins(0)
    ax.tick_params(axis="both", bottom=0, left=0, labelbottom=0,labelleft=0)
    # ADD TEXT IN THE COLOR BARS #
    for idx,x in enumerate(labels):
        align = (x[0] + x[1])/2
        axb.text(align,.5,names_abbrev[idx], fontsize=9, rotation=90, horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[PathEffects.withStroke(linewidth=.5, foreground="w")])
        axl.text(.5,align,names_abbrev[idx], fontsize=9, horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[PathEffects.withStroke(linewidth=.5, foreground="w")])
    fig.set_size_inches(30,18)

    # plt.savefig(mat_path.replace('.csv','.png'), dpi=1200, format='png', bbox_inches='tight')
    # CLEAR FIGURE #
    # plt.clf()
    # plt.cla()
    # plt.close()


def get_flat_inds_for_net(net, within=False):
    range_list = [(0,24),(24,64), (64,69), (69,110),(110,142),(142,166),(166,213),
        (213, 221),(221, 225),(225, 263),(263, 271),(271, 294),(294, 333),(333, 353)]
    names_abbrev = ['Auditory','CingOperc','CingPar','Default','DorsalAtt','FrontoPar','None',
        'RetroTemp','Salience','SMhand','SMmouth','VentralAtt','Visual','Subcort']    
    net_ind = names_abbrev.index(net)
    net_start, net_end= range_list[net_ind]

    mask = np.ones((352,352))
    mask[:, net_start:net_end+1] = 2
    mask[net_start:net_end+1, :] = 2

    if within == True:
        mask = np.ones((352,352))
        mask[net_start:net_end+1, net_start:net_end+1] = 2

    flatmask = np.triu(mask).flatten()[np.triu(mask).flatten().nonzero()]
    return np.where(flatmask == 2)[0]


def get_flat_inds_for_block(net1,net2):

    # t = np.zeros(62128)
    # t[get_flat_inds_for_block('Default','Default')] = 1
    # make_corrfig(triangularizeweights(t))
    
    range_list = [(0,24),(24,64), (64,69), (69,110),(110,142),(142,166),(166,213),
        (213, 221),(221, 225),(225, 263),(263, 271),(271, 294),(294, 333),(333, 353)]
    
    names_abbrev = ['Auditory','CingOperc','CingPar','Default','DorsalAtt','FrontoPar','None',
        'RetroTemp','Salience','SMhand','SMmouth','VentralAtt','Visual','Subcort']
    
    
    net1_ind = names_abbrev.index(net1)
    net2_ind = names_abbrev.index(net2)

    net1_start, net1_end = range_list[net1_ind]
    net2_start, net2_end = range_list[net2_ind]

    mask = np.ones((352,352))
    mask[net1_start:net1_end, net2_start:net2_end+1] = 2
    mask[net2_start:net2_end, net1_start:net1_end+1] = 2
    

    flatmask = np.triu(mask).flatten()[np.triu(mask).flatten().nonzero()]
    return np.where(flatmask == 2)[0]

def loocv_svm(X,y):
    cv = LeaveOneOut()
    clf = svm.SVC(kernel='linear', C=1, random_state=1)
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-3)
    print('LOOCV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    return scores

def train_test_svm(train_x, train_y, test_x, test_y):
    clf = svm.SVC(kernel='linear', C=1, random_state=1)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)
    scores = (predictions==test_y)
    print(clf.score(test_x, test_y))
    return scores