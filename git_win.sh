clear

git add .

my_date=$(date)

echo $my_date

git commit -m "Windows, Checkin Timestamp::$my_date"

git push origin main
