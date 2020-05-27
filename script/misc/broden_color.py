from segmenter import get_segmenter
from lib.netdissect.segviz import segment_visualization, segment_visualization_single, high_contrast

external_model = get_segmenter("bedroom")
labels, cats = external_model.get_label_and_category_names()
labels = labels[0]
colors = [high_contrast[(i+1)%len(high_contrast)]
    for i in len(labels)]