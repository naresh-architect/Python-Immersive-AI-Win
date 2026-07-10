Clear-Host

git add .

$my_date = Get-Date

Write-Output $my_date

git commit -m "Windows, Checkin Timestamp::$my_date"

git push origin main
