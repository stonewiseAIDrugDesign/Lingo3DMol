# python inference/inference_multiproces_ed_forwenzhang.py \
python inference/inference_avoid_clash.py \
--cuda '0' --cuda_list 0 \
--input_list datasets/dude_files \
--savedir try_out \
--coc_dis 2.5 --nci_thrs 0.7 --frag_len 15 --tempture 1.0 --nci_choose_thred 0.0 --topk 5  --USE_THRESHOLD \
--caption_path checkpoint/gen_mol.pkl \
--isMultiSample  --isGuideSample --saveMol --neednum 8 --gennums 100 --runtime 1 \
--sample_num 40 --epoch 100000 --prod_time 1
