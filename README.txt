I. Introduction:																				 		   						
Project Title:			Handwritten Math Expressions Recognition using
						a Recursive approach.
Team Members:			Bhavin Bhuta (bsb5375@rit.edu),                                   	   						
						Dhaval Chauhan (dmc8686@rit.edu),                                 
								Computer Science Department	   				
								Rochester Institute of Technology  				
Developed in: 			Python							   					
Additional Toolkits: 	Numpy, Scipy, Scikit-Learn	                  



II. Instructions:
1) Parser file (project3.py) takes two parameters that is the input textfile's full path,
	and the type of parsing you want to do, i.e. 1) Symbol level, 2) Stroke level

	For symbol level parsing(uses GT symbols):
	python project3.py "D:\Some Directory\Inside Directory\Other Sub Dir\FinalSubDir\testtextfile.txt" 1 
	
	For stroke level parsing:
	python project3.py "D:\Some Directory\Inside Directory\Other Sub Dir\FinalSubDir\testtextfile.txt" 2
	
	Note: full path is required so that the program can extract directory of the Textfile from the
			input path and use that to save the output LG file folder. The textfile must contain full 
			paths of the INKML files listed one below the other.
			
2) A folder gets saved with the LG files of the INKML files that were in the input textfile
	in the directory of the input textfile. A confirmation message is displayed as well as shown below
	
	"Output folder for input txtfile: test.txt created in the same folder by the name: test_strokelevel"



	





