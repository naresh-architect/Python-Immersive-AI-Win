echo "Adding all files to git staging..."
sleep 1
git add .

echo "Capturing current timestamp..."
sleep 1
my_date=$(date)
echo "Timestamp: $my_date"

sleep 1
echo "Creating git commit with timestamp..."
git commit -m "Macbook Pro, Checkin Timestamp::$my_date"

sleep 1
echo "Pushing changes to origin main..."
git push origin main

sleep 1
echo "done"