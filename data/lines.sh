dirs_=`ls forecast`
for dir_ in $dirs_
do
	echo 'forecast/'+$dir_
	cat 'forecast/'$dir_ | wc -l
	#cat $dir_ | head -2 
	#cat $dir_ | tail -1
	#echo "______________________________________________________"
done
