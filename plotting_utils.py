import cbayes.distributions
import ipywidgets
import numpy as np
# import seaborn as sb
from matplotlib import pyplot as plt
import scipy.stats as ss
import nbinteract as nbi

primary_color = '#bdd363'
secondary_color = '#60605d'

dirname, fname = '', ''

def plot_results(output, 
                xlabel="Dollars",
                s="Realized Gifts",
                max_percentile = 100,
                show_range = False,
                scaling_x = 1,
                bins  = 1000,
                save=False):

    G = ss.gaussian_kde(output)
    mn = np.percentile(output,0)
    
    mx = np.percentile(output,max_percentile)
    if max_percentile != 100:
        print('Note:\t %1.1f%% chance that result is between'%(100-max_percentile), '%2.3e and %2.3e'%( mx, np.percentile(output,100) ) )
    x = np.linspace(mn-0.05,mx+0.05,bins)
    y = G.pdf(x)

    plt.figure(figsize=(20,10))
    plt.plot(x,y, color=primary_color)

    m = np.mean(output)
    z = G.pdf([m])

    plt.scatter(m, G.pdf([m]), marker='o', s=100)
    plt.vlines(m, 0, z, lw=1, color='black')
    plt.annotate(' Mean: %2.2e'%(m/scaling_x), [m*1.005,z*.75], fontsize=18, clip_on=True, rotation=-90)

    # IF THE MEDIAN and MEAN DIFFER BY LESS THAN %tage OF RANGE, DO NOT PLOT
    if np.abs(m - np.percentile(output,50))/(mx-mn) < 0.025:
        q_show = [25, 75]
    else:
        q_show = [25, 50, 75]

    for q in q_show:
        mm = np.percentile(output,q)
        plt.vlines(mm, 0, z, lw=1, color='black')
        plt.annotate(' Q%d: %2.2e'%(q,mm/scaling_x), [mm*1.005,0.575*z], fontsize=14, clip_on=True, rotation=-90)


    a = np.argmax(y)
    max_val = y[a]
    plt.scatter([x[a]], [y[a]], marker='o', s=100)
    plt.annotate('Most Likely: %2.2e'%(x[a]/scaling_x), [x[a], 1.01*y[a]], fontsize=18, clip_on=True)
    if show_range:
        plt.annotate(' Min: %2.2e'%(mn/scaling_x), [mn,0.95*z], fontsize=18, clip_on=True, rotation=-90)
        plt.annotate(' Max: %2.2e'%(mx/scaling_x), [mx,0.95*z], fontsize=18, clip_on=True, rotation=-90)

    plt.ylabel("Relative Likelihood"+'\n')
    plt.title("\n%s\n"%s)
    ax = plt.gca()
    # Rewrite the  labels
    
    

    x_labels = ax.get_xticks()
    scaling_label = '(thousands)'
    if scaling_x == 1000:
        scaling_label = '(thousands)'
    elif scaling_x == 1E6:
        scaling_label = '(millions)'
    else:
        scaling_label = ''
    
    ax.set_xticklabels(['%2.1f'%(x/scaling_x) for x in x_labels])
    ax.set_yticks(np.linspace(0,max_val,5))
    plt.xlabel('\n'+"%s\n%s"%(xlabel.capitalize(), scaling_label))
    y_labels = ax.get_yticks()
    ax.set_yticklabels(['%2d'%np.ceil(100*y/max_val) for y in y_labels])
    plt.tight_layout()
    if save:
        print('Saving as', '%s/%s_%s.png'%(dirname,fname+'_'+s.lower().replace(' ', '_').replace('\n', ''), estimate_type) )
        plt.savefig('%s/%s_%s.png'%(dirname,fname+'_'+s.lower().replace(' ', '_').replace('\n', ''), estimate_type))
        
    plt.show()


def makesliders(D):
    slider_list = []
#     ax_list = []
    for label in D.keys():
        slider_list.append(ipywidgets.VBox([ipywidgets.Label(label),
#                               ipywidgets.Dropdown(options=list(cbayes.distributions.supported_distributions().keys()))
#                               ipywidgets.FloatSlider(value=0, min=0, max=10, description='loc'),
#                               ipywidgets.FloatSlider(value=1, min=0, max=10, description='scale'),
                                ipywidgets.FloatText(value=0.0, description='loc'),
                                ipywidgets.FloatText(value=1.0, description='scale'),          
                                ipywidgets.FloatSlider(value=1.0, min=0, max=20, description='a'), 
                                ipywidgets.FloatSlider(value=1.0, min=0, max=20, description='b'),
                                ipywidgets.Dropdown(options=['normal', 'beta', 'uniform'], value='beta'),
                                ipywidgets.widgets.Output(),
                                ipywidgets.Button(description='Preview')]))
    return slider_list


def plot_current_dist(T, fname='tempfig', ftype='png', quartiles=True):
    p = T.selected_index
    
    param = T.children[p].children[0].value
    loc = T.children[p].children[1].value
    scale = T.children[p].children[2].value
    a = T.children[p].children[3].value
    b = T.children[p].children[4].value
    d = T.children[p].children[5].value
    f = cbayes.distributions.parametric_dist(0)
    if d == 'beta':
        f.set_dist(dim=0, dist=d, kwds={'a':a, 'b':b, 'loc':loc, 'scale':scale})
    elif d in ['normal', 'uniform']:
        f.set_dist(dim=0, dist=d, kwds={'loc':loc, 'scale':scale})
    n = 250
    p_delta = 0.05*scale # spacer for plotting.
    if d != 'normal':
        x = np.linspace(loc - p_delta - 1*(scale==0), 1*(scale==0) + loc+scale + p_delta, n)
    else:
        x = np.linspace(loc - 3.0*scale, loc + 3.0*scale, n)
    
    
    opts = {
    'animation_duration': 250,
    }

    
#     def x_values(max): 
#         return np.arange(0, max)

#     def y_values(xs, sd):
#     return xs + np.random.normal(0, scale=sd, size=len(xs))
    
    
    
#     fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
#     ax = fig.add_subplot(1,1,1)
#     ax.clear()
    if scale == 0: # if fixed constant,
        y = 0*x
#         plt.vlines(loc,0,1)
        nbi.line([loc, loc], [0,1], options=opts)
    else:
        y = f.pdf(x)
#         ax.plot(x,y, color='cyan')
        nbi.line(x, y, options=opts)
        if quartiles:
            q = f.ppf(np.array([0.25,0.5,0.75]).reshape(-1,1))
#             for z in q:
#                 plt.scatter(z, f.pdf(z) , marker='o', s=50)
#                 plt.vlines(z, 0,f.pdf(z), lw=1, color='k')
#                 plt.annotate(' %2.2f'%z, [z, f.pdf(z)*np.max(y)])
    paramstr = ' '.join([s.capitalize() for s in param.split('_')])
    paramstr = paramstr.replace('Per ', 'per ')
    paramstr = paramstr.replace('To ', 'to ')
    paramstr = paramstr.replace('In ', 'in ')
    
#     plt.xlabel('\n'+paramstr)
#     plt.ylabel("Relative Likelihood"+'\n')
#     plt.tight_layout()
#     plt.savefig(fname+paramstr.lower().replace(' ', '_')+'.'+ftype)
#     plt.show()

    pass


def paramdict(items, params):
    return {i:{j: None for j in params} for i in items}

def assumptions(U, preview=False):
    items = list(U.keys())
    TAB = ipywidgets.Tab([makeTabs(U[k], preview) for k in items])
    for k in range(len(items)):
        TAB.set_title(k, items[k].replace('_', ' '))
        for j in range(len(list(U[items[k]].keys()))):
            paramstr = TAB.children[k].children[j].children[0].value 
            # TAB.children[k].children[j].children[0].value = items[k]+'_'+paramstr # maybe use in another label?
    return TAB

def extractS(T, dim=False):
    S = {}
    for t in range(len(T.children)):
        
        d = T.children[t].children[5].value
        param = T.children[t].children[0].value
        if dim:
            S[param] = {'dist': d, 'kwds': {}, 'dim': t}
        else:
            S[param] = {'dist': d, 'kwds': {}}
        S[param]['kwds']['loc'] = T.children[t].children[1].value
        S[param]['kwds']['scale'] = T.children[t].children[2].value
        if d is 'beta':
            S[param]['kwds']['a'] = T.children[t].children[3].value
            S[param]['kwds']['b'] = T.children[t].children[4].value
    return S

def extract(TAB):
    return {TAB._titles[str(t)].replace(' ', '_') : extractS(TAB.children[t]) for t in range(len(TAB.children))}

def dict_to_array(D):
    return np.concatenate([D[k].reshape(-1,1) for k in D], axis=1)

def makeTabs(S, preview=False):
    slider_list = makesliders(S)
    T = ipywidgets.Tab(slider_list)
    
    for t in range(len(T.children)):
        param = T.children[t].children[0].value
        d = T.children[t].children[5].value
        T.set_title(t, param.replace('_', ' '))
        if S[param] is not None:
            try:
                T.children[t].children[1].value = S[param]['kwds']['loc'] 
                T.children[t].children[2].value = S[param]['kwds']['scale']
            except TypeError:
                T.children[t].children[1].value = 0
                T.children[t].children[2].value = 1
            try: # by using this framework instead of "if/then", we don't have to add code as we add other distributions.
                T.children[t].children[3].value = S[param]['kwds']['a']
                T.children[t].children[4].value = S[param]['kwds']['b']
            except Exception as err:
                # T.children[t].children[3].disabled = True
                # T.children[t].children[4].disabled = True
                pass
            
        plt_prev = T.children[t].children[6]
        
#         @plt_prev.capture(clear_output=True, wait=True)
        def callbck(*kwds):
            tt = T.selected_index
            param = T.children[tt].children[0].value # string
            loc = T.children[tt].children[1].value
            scale = T.children[tt].children[2].value
            a = T.children[tt].children[3].value
            b = T.children[tt].children[4].value
            d = T.children[tt].children[5].value
            f = cbayes.distributions.parametric_dist(0)
            if d == 'beta':
                f.set_dist(dim=0, dist=d, kwds={'a':a, 'b':b, 'loc':loc, 'scale':scale})
                T.children[tt].children[3].disabled = False
                T.children[tt].children[4].disabled = False
            elif d in ['normal', 'uniform']:
                f.set_dist(dim=0, dist=d, kwds={'loc':loc, 'scale':scale})
                T.children[tt].children[3].disabled = True
                T.children[tt].children[4].disabled = True
            n = 201
            p_delta = 0.05*scale # spacer for plotting.
            if d != 'normal':
                x = np.linspace(loc - p_delta - 1*(scale==0), 1*(scale==0) + loc+scale + p_delta, n)
            else:
                x = np.linspace(loc - 3.0*scale, loc + 3.0*scale, n)
            
            opts = {
            'animation_duration': 250,
            }



        #     fig = plt.figure()
            # ax = fig.add_axes([0,0,1,1])
        #     ax = fig.add_subplot(1,1,1)
        #     ax.clear()
            if scale == 0: # if fixed constant,
                y = 0*x
        #         plt.vlines(loc,0,1)
                nbi.line([loc, loc], [0,1], options=opts)
            else:
                y = f.pdf(x)
        #         ax.plot(x,y, color='cyan')
                def y_values(): return y
                def x_values(): return x
                nbi.line(x_values, y_values, options=opts)
#                 if quartiles:
#                     q = f.ppf(np.array([0.25,0.5,0.75]).reshape(-1,1))
        #             for z in q:
        #                 plt.scatter(z, f.pdf(z) , marker='o', s=50)
        #                 plt.vlines(z, 0,f.pdf(z), lw=1, color='k')
        #                 plt.annotate(' %2.2f'%z, [z, f.pdf(z)*np.max(y)])
#             fig = plt.figure()
#             ax = fig.add_axes([0,0,1,1])
# #             ax = fig.add_subplot(1,1,1)
#             #     ax.clear()
#             if scale == 0: # if fixed constant,
#                 y = 0*x
#                 plt.vlines(loc,0,1)
#             else:
#                 y = f.pdf(x)
#                 ax.plot(x,y, color='cyan')
#                 q = f.ppf(np.array([0.25,0.5,0.75]).reshape(-1,1))
#                 quartiles = True # TODO: add this to the tab as an option.
#                 if quartiles:
#                     for z in q:
#                         plt.scatter(z, f.pdf(z) ,marker='o', s=50)
#                         plt.vlines(z, 0,f.pdf(z), lw=1, color='k')
#                         plt.annotate(' %2.3f'%(z), [z, f.pdf(z)])
#             paramstr = ' '.join([s.capitalize() for s in param.split('_')])
#             paramstr = paramstr.replace('Per ', 'per ')
#             paramstr = paramstr.replace('To ', 'to ')
#             plt.xlabel('\n'+paramstr)
#             plt.ylabel("Relative Likelihood"+'\n')
#             plt.show()
            
            pass
        T.children[t].children[7].on_click(callbck) # assign callback to the preview button
        T.selected_index = t # these two lines render initial plots of all assumptions. 
        if preview:
            callbck() # make plots. in addition to this visual aid, this approach avoids later resizing of tabs once the button is pressed
    return T
















def pltdata(data, view_dim_1=0, view_dim_2=1, eta_r=None, inds=None, N=None,  color="eggplant", space=0.05, svd=False): # plots first N of accepted, any 2D marginals specified
    if type(data) is np.ndarray:

        if inds is not None:
            data_subset = data[inds,:]
        else:
            data_subset = data
        if N is not None:
            data_subset = data_subset[0:N]
    else:
        try: # try to infer the dimension... 
            d = len(data.rvs())
        except TypeError:
            try:
                d = data.rvs().shape[1]
            except IndexError:
                d = 1
         
        data = data.rvs((N,d)) # if we get a distribution object, use it to generate samples.  
        data_subset = data 
    x_data = data_subset[:, view_dim_1]
    try:
        y_data = data_subset[:, view_dim_2]
    except IndexError:
        y_data = x_data
    rgb_color = sb.xkcd_rgb[color]

    if view_dim_1 == view_dim_2:
        sb.kdeplot(x_data, color=rgb_color)
        if eta_r is not None:
            plt.figure()
            plt.scatter(data[:,view_dim_1], eta_r, alpha=0.1, color=rgb_color)
    else:
            # perform SVD and show secondary plot
        if svd:
            offset = np.mean(data_subset, axis=0)
            la = data_subset - np.array(offset)
            U,S,V = np.linalg.svd(la)
            new_data = np.dot(V, la.transpose()).transpose() + offset
            x_data_svd = new_data[:,view_dim_1]
            y_data_svd = new_data[:,view_dim_2]
            
            sb.jointplot(x=x_data_svd, y=y_data_svd, kind='kde', 
                         color=rgb_color, space=space, stat_func=None)
        
        else: # no SVD - show scatter plot
            plt.figure()
            if inds is None:
                plt.scatter(data[0:N,view_dim_1], data[0:N,view_dim_2], alpha=0.2, color=rgb_color)
                
            else:
                plt.scatter(data[inds[0:N],view_dim_1], data[inds[0:N],view_dim_2], alpha=0.2, color=rgb_color)
                # plt.axis('equal')
                min_1 = np.min(data[:,view_dim_1])
                min_2 = np.min(data[:,view_dim_2])
                max_1 = np.max(data[:,view_dim_1])
                max_2 = np.max(data[:,view_dim_2])
                plt.xlim([min_1, max_1])
                plt.ylim([min_2, max_2])
        sb.jointplot(x=x_data, y=y_data, kind='kde', 
                     color=rgb_color, space=space, stat_func=None)
        
    plt.show()
