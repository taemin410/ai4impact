files_=`ls forecast`
cd forecast
for file in $files_
do
	echo $file
	cat $file | tail -n +4 > $file
done
