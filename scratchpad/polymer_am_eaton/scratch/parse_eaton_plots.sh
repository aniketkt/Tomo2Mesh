#!/bin/bash

python plot_eaton_graphs_ae.py "1"
# python plot_eaton_graphs_ae.py "9"
# python plot_eaton_graphs_ae.py "2"
# python plot_eaton_graphs_ae.py "3"
# python plot_eaton_graphs_ae.py "4"
# python plot_eaton_graphs_ae.py "5"
# python plot_eaton_graphs_ae.py "6"
# python plot_eaton_graphs_ae.py "7"
# python plot_eaton_graphs_ae.py "8"
# python plot_eaton_graphs_ae.py "10"
# python plot_eaton_graphs_ae.py "11"
# python plot_eaton_graphs_ae.py "12"
# python plot_eaton_graphs_ae.py "injmold"

# for val in {1..12}; do
    # if [[$val -eq 1]]; then
    #     start_layer = 1
    #     end_layer = 4
    #     sample_tag = "1"
    # elif [[$val -eq 2 || val -eq 3]]; then
    #     start_layer = 1
    #     end_layer = 5
    #     sample_tag = val
    # elif [[$val -eq 4 || $val -eq 6 || $val -eq 8 || $val -eq 10]]; then
    #     start_layer = 1
    #     end_layer = 4
    #     if [[$val -eq 4]]; then
    #         sample_tag = "45"
    #     elif [[$val -eq 6]]; then
    #         sample_tag = "67"
    #     elif [[$val -eq 8]]; then
    #         sample_tag = "89"
    #     elif [[$val -eq 10]]; then
    #         sample_tag = "1011"
    #     fi
    # elif [[$val -eq 5 || $val -eq 7 || $val -eq 9 || $val -eq 11]]; then
    #     start_layer = 7
    #     end_layer = 10
    #     if [[$val -eq 5]]; then
    #         sample_tag = "45"
    #     elif [[$val -eq 7]]; then
    #         sample_tag = "67"
    #     elif [[$val -eq 9]]; then
    #         sample_tag = "89"
    #     elif [[$val -eq 11]]; then
    #         sample_tag = "1011"
    #     fi
    # elif [[$val -eq 12]]; then
    #     start_layer = 1
    #     end_layer = 3
    #     sample_tag = "12"
    # fi

    # python plot_eaton_graphs.py "$val" "$start_layer" "$end_layer" "$sample_tag"
#     echo $val
# done

