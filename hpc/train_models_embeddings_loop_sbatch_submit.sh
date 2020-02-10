cd $AUTOREVIEW_PROJECT_DIR
# DATA_SUBDIR=data/hpc/reviews_wos_collect_smallsize_sample_20200114
DATA_SUBDIR=data/hpc/reviews_wos_collect_largesize_sample_20200114
# ID_LIST=data/reviews_wos_ids_smallsize_sample_20200114.tsv
ID_LIST=data/reviews_wos_ids_largesize_sample_20200114.tsv
for SEED_SIZE in 200; do
	for TRANSFORMER_NUM in 10 11; do
		echo "Submitting for TRANSFORMER_NUM ${TRANSFORMER_NUM} SEED_SIZE ${SEED_SIZE}"
		for (( i = 0; i < 100; i++ )); do
			sbatch -p ckpt -A stf-ckpt --mail-user $EMAIL --chdir $AUTOREVIEW_PROJECT_DIR --output $DATA_SUBDIR/seed_papers_size_${SEED_SIZE}/select_wos_id_and_train_models_${i}_features${TRANSFORMER_NUM}.out hpc/batch_scripts/select_wos_id_and_train_models_embeddings.sbatch $ID_LIST $i $DATA_SUBDIR/seed_papers_size_${SEED_SIZE}/ $TRANSFORMER_NUM /gscratch/stf/jporteno/wos_201912/hpc/data/title_embeddings/
		done
		echo ""
	done
done
