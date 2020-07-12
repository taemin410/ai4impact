files_=`ls`
for file in $files_
do
	echo $file
	cat $file | tail -n +4 > "../history_cleaned/$file"
done
