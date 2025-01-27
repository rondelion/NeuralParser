BEGIN{sum=0}
{sum = sum + $3}
END{print sum, FNR}