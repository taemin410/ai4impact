cat wind_energy_v2.csv | head -1000 > ../test_data/wind_energy_v2.csv
cd history_cleaned
dirs_=`ls`
for dir_ in $dirs_
do
	cat $dir_ | head -167 > ../../test_data/history_cleaned/$dir_
done
