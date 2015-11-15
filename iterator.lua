--[[ iterator in lua--]]

function square(iteratorMaxCount, currentNumber)
	if currentNumber<iteratorMaxCount
	then
		currentNumber = currentNumber + 1
		print("currentNumber: ", currentNumber,"\n")
	return currentNumber, currentNumber*currentNumber
	end
end

function squares(iteratorMaxCount)
	print("in squares \n")
	return square, iteratorMaxCount,0
end

function printSquares(t)
	for i,n in squares(t)
	do 
		print(i,n)
	end
end