dirs_=`ls history_cleaned`
for dir_ in $dirs_
do
	echo 'history_cleaned/'+$dir_
	cat 'history_cleaned/'+$dir_ | wc -l
	#cat $dir_ | head -2 
	#cat $dir_ | tail -1
	#echo "______________________________________________________"
done
