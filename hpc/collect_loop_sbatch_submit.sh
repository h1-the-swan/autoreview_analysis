cd $AUTOREVIEW_PROJECT_DIR
for SEED_SIZE in 200; do
	echo "Submitting for SEED_SIZE ${SEED_SIZE}"
	for (( i = 0; i < 100; i++ )); do
		sbatch -p ckpt -A stf-ckpt --mail-user $EMAIL --output data/hpc/reviews_wos_collect_largesize_sample_20200114/seed_papers_size_${SEED_SIZE}/select_wos_id_and_collect_${i}.out hpc/batch_scripts/select_wos_id_and_collect_nospark_different_seed_sizes.sbatch data/reviews_wos_ids_largesize_sample_20200114.tsv $i data/hpc/reviews_wos_collect_largesize_sample_20200114/seed_papers_size_${SEED_SIZE}/ $SEED_SIZE
	done
	echo ""
done
