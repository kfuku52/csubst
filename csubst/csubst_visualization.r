mode = ifelse(length(commandArgs(trailingOnly=TRUE))==1, 'debug', 'batch')

library(Biostrings)
library(ape)
library(ggplot2)
library(ggtree)
library(gridExtra)
library(rkftools)
library(ggmsa)

options(stringsAsFactors=FALSE)
#options(warn=-1)
options(repr.matrix.max.cols=200)
#options(repr.matrix.max.rows=600, repr.matrix.max.cols=200)

font_size_factor = 0.352777778

if (mode=="debug") {
    dir_csubst = '/Volumes/kfT7/Dropbox/collaborators/Toshiya_Ando/20230401_insect_coloration/HOG0005974_cuticle_protein'
    #dir_csubst = getwd()
    setwd(dir_csubst)
    args = c()
    args = c(args, paste0('--dir_csubst=', dir_csubst))
} else if (mode=="batch") {
    args = commandArgs(trailingOnly=TRUE)
}

cat('arguments:\n')
args = rkftools::get_parsed_args(args, print=TRUE)

add_numerical_node_labels = function(phylo, rename_node=FALSE) {
    if (rename_node) {
        phylo[['node.label']] = paste0('n', 1:length(tree[['node.label']]))
    }
    all_leaf_names = phylo[['tip.label']]
    all_leaf_names = sort(all_leaf_names)
    leaf_numerical_labels = list()
    for (i in 0:length(all_leaf_names)) {
        leaf_numerical_labels[all_leaf_names[i]] = 2**i
    }
    intnode_numerical_labels = list()
    for (st in subtrees(phylo)) {
        leaf_names = st[['tip.label']]
        numerical_label = sum(unlist(leaf_numerical_labels[leaf_names]))
        intnode_numerical_labels[[st[['node.label']][1]]] = numerical_label
    }
    numerical_labels = c(unlist(leaf_numerical_labels), unlist(intnode_numerical_labels))
    numerical_labels = sort(numerical_labels)
    names = names(numerical_labels)
    numerical_labels = (1:length(numerical_labels)) - 1
    names(numerical_labels) = names
    attr(phylo, 'numerical_label') = c(numerical_labels[phylo[['tip.label']]], numerical_labels[phylo[['node.label']]])
    return(phylo)
}

get_line_coordinate = function(g, numerical_labels, jitter=FALSE, pairwise=FALSE) {
    if (pairwise) {
        nl_combinations = combn(numerical_labels, 2)
        dat = data.frame(nl_start=nl_combinations[1,], nl_end=nl_combinations[2,])
        for (attr in c('start','end')) {
            dat = merge(dat, g$data[c('numerical_label','branch','y')], all.x=TRUE, by.x=paste0('nl_',attr), by.y='numerical_label')
            colnames(dat) = sub('^branch$', paste0('x_',attr), colnames(dat))
            colnames(dat) = sub('^y$', paste0('y_',attr), colnames(dat))
        }
        out = dat
    } else {
        is_target = (g[['data']][['numerical_label']] %in% numerical_labels)
        dat = g[['data']][is_target,]
        out = dat[,c('numerical_label','branch','y')]
        colnames(out) = sub('^y$', 'y_start', colnames(out))
        rownames(out) = NULL
        out = out[order(out[['y_start']], decreasing=TRUE),]
        out[,'x_start'] = out[['branch']]
        if (jitter) {
            #amount = 
            out[,'x_start'] = jitter(out[['x_start']], amount=amount)
        }
        out[,'x_end'] = out[(1:nrow(out))+1,'x_start']
        out[,'y_end'] = out[(1:nrow(out))+1,'y_start']
        out = na.omit(out)
        rownames(out) = NULL
    }
    return(out)
}

density_scatter = function(x1,x2,
                               ylim=c(min(x2),max(x2)),
                               xlim=c(min(x1),max(x1)),
                               xlab="",ylab="",main="") {
    # http://knowledge-forlife.com/r-color-scatterplot-points-density/
    df <- data.frame(x1,x2)
    x <- densCols(x1,x2, colramp=colorRampPalette(c("black", "white")))
    df$dens <- col2rgb(x)[1,] + 1L
    cols <-  colorRampPalette(c("#000099", "#00FEFF", "#45FE4F","#FCFF00", "#FF9400", "#FF3100"))(256)
    df$col <- cols[df$dens]
    plot(x2~x1, data=df[order(df$dens),], 
         ylim=ylim,xlim=xlim,pch=20,col=col,
         cex=2,xlab=xlab,ylab=ylab, las=1,
         main=main)
}

get_cb = function(dir_csubst, arity, subsample=FALSE) {
    csubst_files = list.files(dir_csubst)
    if (subsample) {
        kwd = 'csubst_cb_subsample_[0-9].*'
    } else {
        kwd = 'csubst_cb_[0-9].*'
    }
    cb_files = csubst_files[grep(kwd, csubst_files)]
    file_path = paste0(dir_csubst, cb_files[grep(as.character(arity), cb_files)])
    cb = read.table(file_path, sep='\t', header=TRUE)
    return(cb)
}

overlay_convergence = function(g, cb, stat='OCNany2spe_dev', top_percent_to_show, max_num_to_show, is_target_only, show_label=TRUE) {
    cb[,paste0(stat, '_rank')] = rank(-cb[[stat]], na.last=TRUE, ties.method="first")
    bid_cols = colnames(cb)[grep('branch_id_', colnames(cb))]
    arity = length(bid_cols)
    if (is_target_only) {
        cb_target = cb[(cb[,'branch_num_fg']==arity),]
    } else {
        cb_target = cb
    }
    num_show = floor(nrow(cb_target) * top_percent_to_show * 0.01)
    num_show = max(num_show, 1)
    cat(paste0(top_percent_to_show, '% (', num_show, '+tiers/', nrow(cb), ') of top protein convergence will be analyzed.\n'))
    cb_target = cb_target[rev(order(cb_target[,stat])),]
    cb_target = cb_target[(cb_target[,stat]>0),]
    rownames(cb_target) = NULL    
    cat('There are', nrow(cb_target), 'branch combinations that satisfy', stat, '> 0\n')
    threshold = cb_target[num_show,stat]
    cb_target = cb_target[(cb_target[,stat]>=threshold),]
    if ((nrow(cb_target)>max_num_to_show)&(!is.infinite(threshold))) {
        cat('For visualization, top protein convergence will be analyzed if branch combinations have', stat, 'equal to the value in', max_num_to_show, 'th ranked one.\n')
        threshold = cb_target[max_num_to_show,stat]
        if (threshold==Inf) {
            threshold = max(cb_target[(cb_target[,stat]!=Inf),stat])
        }
        cb_target = cb_target[(cb_target[,stat]>threshold),]
    }
    cat('There are', nrow(cb_target), 'branch combinations that satisfy', stat, '>', threshold, '\n')
    counts = c()
    if (nrow(cb_target)>0) {
        df_line_coords = data.frame()
        for (i in 1:nrow(cb_target)) {
            numerical_labels = unlist(cb_target[i,bid_cols])
            line_color = ifelse(cb_target[i,'is_fg']=='Y', 'firebrick', 'gray50')
            line_coords = get_line_coordinate(g, numerical_labels)
            line_coords[1,'OCNany2spe'] = round(cb_target[i,'OCNany2spe'], digits=1)
            line_coords[1,stat] = round(cb_target[i,stat], digits=1)
            line_coords[1,paste0(stat, '_rank')] = cb_target[i,paste0(stat, '_rank')]
            g = g + geom_curve(aes(x=x_start, y=y_start, xend=x_end, yend=y_end, alpha=0.1), 
                                                   size=0.3, curvature=jitter(0,amount=0.05), colour=line_color, 
                                                   data=line_coords, show.legend=FALSE)
            counts = c(counts, unname(numerical_labels))
            df_line_coords = rbind(df_line_coords, line_coords)
        }
        labels = paste0(df_line_coords[,'OCNany2spe'], '/', df_line_coords[,stat], '/', df_line_coords[,paste0(stat, '_rank')])
        cat(paste0('Branch annotations: OCNany2spe/', stat, '/rank\n'))
        df_label_coords = data.frame(
            'x' = df_line_coords[,'x_start'] + ((df_line_coords[,'x_end'] - df_line_coords[,'x_start'])/2),
            'y' = df_line_coords[,'y_start'] + ((df_line_coords[,'y_end'] - df_line_coords[,'y_start'])/2),
            label = labels
        )
        if (show_label) {
            g = g + ggrepel::geom_text_repel(mapping=aes(x=x, y=y, label=label), data=df_label_coords,
                                                                 color='firebrick', size=2, hjust=0.5, vjust=0.5, force=0.02)
        }
    }
    counts = table(counts)
    circle_coords = data.frame(numerical_label=as.numeric(names(counts)))
    circle_coords[,'count'] = as.numeric(unname(counts))
    g[['data']] = merge(g[['data']], circle_coords, by='numerical_label', sort=FALSE, all.x=TRUE)
    g[['data']][(is.na(g[['data']][,'count'])),'count'] = 0
    g[['data']][,'show_circle'] = (g[['data']][,'count']>0)
    #g[['data']][,'is_fg'] = factor(g[['data']][['is_fg']], levels=rev(unique(g[['data']][['is_fg']])))
    g = g + geom_point2(aes(x=branch, y=y, subset=show_circle, color=!is_fg), shape=16, size=2.5, show.legend=FALSE)
    g = g + geom_text2(aes(x=branch, y=y, label=count, subset=show_circle), color='white', size=2, show.legend=FALSE)
    g = g + guides(alpha=FALSE)
    return(g)
}

extract_numerical_node_labels = function(tree) {
    labels = c(tree[['tip.label']], tree[['node.label']])
    numerical_labels = as.integer(sub('.*\\|([0-9]*)$', '\\1', labels))
    numerical_labels[is.na(numerical_labels)] = -1
    label_names = sub('(.*)\\|[0-9]*$', '\\1', labels)
    tip_names = label_names[1:length(tree[['tip.label']])]
    node_names = label_names[(length(tree[['tip.label']])+1):length(label_names)]
    tree[['tip.label']] = tip_names
    tree[['node.label']] = node_names
    attr(tree, 'numerical_label') = numerical_labels
    return(tree)
}

annotate_fg_tip = function(df=g$data, fg_file=file.path(dir_csubst, 'foreground.txt')) {
    df_fg = read.table(fg_file, sep='\t', header=FALSE)
    colnames(df_fg) = c('lineage_num', 'regex')
    df[,'is_fg_tip'] = FALSE
    for (regex in df_fg[['regex']]) {
        is_target_tip = grepl(regex, df[['label']])
        df[is_target_tip,'is_fg_tip'] = TRUE
    }
    return(df)
}

get_alignment_data = function(dir_csubst, b, alignment_min_substitution) {
    fg_subs = unlist(sapply(b[(b[['is_fg']]=='yes'),][['N_sitewise']], function(x){strsplit(x, ',')}))
    names(fg_subs) = NULL
    fg_subs = gsub("^.{1}", "", fg_subs)
    df_fg_subs = data.frame(table(fg_subs))
    df_fg_subs[['site']] = as.integer(gsub(".{1}$", "", df_fg_subs[['fg_subs']]))
    df_fg_subs_show = df_fg_subs[(df_fg_subs[['Freq']]>=alignment_min_substitution),]
    aln_path = file.path(dir_csubst, 'csubst_alignment_aa.fa')
    aln = Biostrings::readAAStringSet(aln_path)
    data2 = tidy_msa(aln)
    data2[['name']] = sub('\\|.*', '', as.character(data2[['name']]))
    is_site = (data2[['position']] %in% df_fg_subs_show[['site']])
    is_tip = !grepl('^n[0-9]+$', as.character(data2[['name']]))
    data2 = data2[(is_site&is_tip),]
    data2[['original_position']] = data2[['position']]
    site_counter = 0
    for (position in unique(data2[['position']])) {
        site_counter = site_counter + 1
        data2[(data2[['original_position']]==position),'position'] = site_counter
    }
    rownames(data2) = NULL
    return(data2)
}

csubst_files = list.files(args[['dir_csubst']])
cb_files = csubst_files[grep('csubst_cb_[0-9].tsv', csubst_files)]
cb_stat_file = csubst_files[grep('csubst_cb_stats.*', csubst_files)]

tree = read.tree(file.path(args[['dir_csubst']], 'csubst_tree.nwk'))
#tree = add_numerical_node_labels(tree)
tree = extract_numerical_node_labels(tree)
b = read.table(file.path(dir_csubst, 'csubst_b.tsv'), sep='\t', header=TRUE)

options(warn=-1)
font_size = 8
branch_colors = c('firebrick', 'gray50')
min_OCNCoD = 0
min_OCNany2any = 0
min_OCSany2any = 0
min_OCNany2spe = 1.5
min_OCSany2spe = 0
min_omegaCany2spe = 3.0
arity_min = 3
stat='omegaCany2spe'
arity_max = 8
foreground_only = TRUE

df_sub = data.frame()
for (arity in seq(arity_min, arity_max)) {
    cat('Arity =', arity, '\n')
    file_path = file.path(dir_csubst, paste0('csubst_cb_', arity, '.tsv'))
    if (!file.exists(file_path)) {
        next
    }
    cb = read.table(file_path, sep='\t', header=TRUE)

    cb2 = cb
    if (foreground_only) {
        cb2 = cb2[(cb2[['is_fg']]=='Y'),]
    }
    conditions = (cb2[['OCNCoD']]>=min_OCNCoD)
    conditions[is.na(conditions)] = FALSE
    cb2 = cb2[(conditions),]
    conditions = (cb2[['OCNany2spe']]>=min_OCNany2spe)
    conditions[is.na(conditions)] = FALSE
    cb2 = cb2[(conditions),]
    conditions = (cb2[['OCSany2spe']]>=min_OCSany2spe)
    conditions[is.na(conditions)] = FALSE
    cb2 = cb2[(conditions),]
    conditions = (cb2[['omegaCany2spe']]>=min_omegaCany2spe)
    conditions[is.na(conditions)] = FALSE
    cb2 = cb2[(conditions),]

    for (fg in c('Y','N')) {
        is_it = (cb2[['is_fg']]==fg)
        if (sum(is_it)==0) {
            cat(paste('no data for arity =', arity, 'fg =', fg, '\n'))
            next
        }
        for (col in c('OCNany2spe','OCSany2spe','OCNany2dif','dSCany2spe','dNCany2spe')) {
            values = cb2[is_it,col]
            tmp = data.frame(
                'k'=arity,
                'fg'=fg,
                'stat'=col,
                'value'=values,
                stringsAsFactors=FALSE
            )
            df_sub = rbind(df_sub, tmp)
        }
    }    
    
    options(repr.plot.width=8, repr.plot.height=6)
    g = ggtree::ggtree(tree, layout='rectangular')
    g[['data']][,'numerical_label'] = as.integer(g[['data']][['numerical_label']])
    g$data$label = unname(sapply(g$data$label, function(x){strsplit(x, split='|', fixed=TRUE)[[1]][1]}))
    g$data = annotate_fg_tip(df=g$data, fg_file=file.path(dir_csubst, 'foreground.txt'))
    g$data = merge(g$data, b, by.x='numerical_label', by.y='branch_id', how='left')
    g[['data']][,'is_fg'] = (g[['data']][,'is_fg']=='yes')
    g[['data']][,'is_mg'] = (g[['data']][,'is_mg']=='yes')
    if (sum(g[['data']][['is_fg']])==nrow(g[['data']])) {
        branch_colors = c('black')
    }
    data_aln = get_alignment_data(dir_csubst, b, alignment_min_substitution=arity)
    data_aln = merge(data_aln, g[['data']][,c('label','is_fg_tip')], by.x='name', by.y='label', how='left')
    
    #g = g + ggplot2::xlim(0, max(g$data$x)+0.5)
    g = g + scale_color_manual(values=branch_colors)
    g = g + theme_tree()
    g = g + geom_treescale(x=0, y=0, offset=1, fontsize=font_size*font_size_factor, linesize=1)
    g = g + ggtree::geom_tree(mapping=aes(color=!is_fg), show.legend=FALSE)
    g = g + ggtree::geom_tiplab(aes(color=!is_fg_tip), size=font_size*font_size_factor, align=FALSE, linetype="dotted", linesize=0.5, show.legend=FALSE)
    g = overlay_convergence(g, cb2, stat=stat, top_percent_to_show=100, max_num_to_show=Inf, is_target_only=FALSE, show_label=FALSE)
    g = g + xlim_tree(3)
    g = g + geom_facet(geom=geom_msa, data=data_aln,  panel='Site', font=NULL, color="LETTER")
    g = g + geom_facet(geom=geom_text, data=data_aln, panel='Site', mapping=aes(x=position, label=character), size=font_size*font_size_factor, color='white')
    g = g + theme(
        strip.text=element_text(size=font_size),
    )

    visualized_sites = unique(data_aln[['original_position']])
    cat('Visualized amino acid sites:', paste(visualized_sites, collapse=', '), '\n')
    num_site = length(visualized_sites)
    facet_widths(g, widths = c(1, num_site/40))

    height = length(tree[['tip.label']]) / 8
    file_name = paste0('csubst_tree.K', arity, '.pdf')
    ggsave(file_name, height=max(3,height), width=7.2)
    g
    cat('\n')
}
cat('Done!\n')


