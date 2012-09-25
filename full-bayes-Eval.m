# the program assumes normal (gaussian) distribution of all the features in the dataset as well as the independence of all the features amongst themselves and builds a decision boundary based classifier to detect spam emails #

# Author:
#	Pranav Gupta 
#	Department of Computer Science and Engineering
#	Indian Institute of Technology Guwahati

#discriminant function in case of normal distribution :
#	g-i(x-bar) = [-1/2 * (x-bar - u-i-bar)' * sigma-i-inverse * (x-bar - u-i-bar)]  - d/2 ln(2*pi)  - 1/2|sigma-i| + ln(P(w-i))
#
#	Assuming the independence of features, we have
#		g-i(x-bar) = [-1/2 * (x-bar - u-i-bar)' * sigma-i-inverse * (x-bar - u-i-bar)]  + ln(P(w-i))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    GLOBAL VARIABLES     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# fetch the data set into a matrix
global D = csvread('spambase.data.txt');

global no_records = rows(D);
global no_classes = 2;
global no_features = columns(D);			# no_features includes the class label
global valFold = 4;							# 4-fold validation as per the requirement of the assignment
global foldSize = floor(rows(D)/valFold);			# size of one fold



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        FUNCTIONS        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################## FUNCTION TO BUILD THE FULL BAYES MODEL #######################
################## ====================================== #######################


function LP = buildModel(M)
	global D valFold foldSize no_features no_classes no_records;

	nRecords = rows(M);
	#printf("number of Records : %d\n", nRecords);
	
	# Calculating values of mu, sigma for all classes
	# ===============================================

	P = zeros(1, no_classes);
	means = zeros(no_features - 1, no_classes);
	vars = zeros(no_features - 1, no_classes);
	
	for class_no = [1:no_classes]
		classData = M(find(M(:,no_features) == class_no),:);
		P(class_no) = rows(classData)/nRecords;
		#printf("classData(%d) : %d\n", class_no, rows(classData));
		for featno = [1:no_features - 1]
			means(featno, class_no) = mean(classData(:,featno));
			vars(featno, class_no) = var(classData(:,featno));
		endfor
	endfor

	LP = [means;vars;P];
	#disp(means);
	#disp(vars);
	disp(P);
endfunction

################# END OF FUNCTION BUILDMODEL() ###################
##################################################################



################## FUNCTION TO CLASSIFY THE TEST DATA #######################
################## ================================== #######################


function CM = detectSpam (M,LP)
	global D valFold foldSize no_features no_classes no_records;
	
	CM = zeros(no_classes);     	# confusion matrix

	# Load the trained Model
	# ======================

	means = LP(1:(no_features - 1),:);
	vars = LP(1:(no_features - 1),:);

	P = LP((2 * no_features - 1),:);

	for rowno = [1:rows(M)]
		rowMfeat = M(rowno,1:(no_features-1));
		rowMclass = M(rowno,no_features);
		
		# calculate the discriminant for each class
		G = zeros(1,no_classes);			# G holds the discriminant function values for all classes

		C = 0;
		maxProb = -inf;				# assign highest discriminant value to a very small number initially
		for class_no = [1:no_classes]
					
			# calculate the discriminant value
			G(class_no) = [(rowMfeat - means(:,class_no)') * inverse(diag(vars(:,class_no))) * (rowMfeat - means(:,class_no)')'] + log(P(class_no)) - 0.5 * log(det(diag(vars(:,class_no))));
			
			#printf("detectSpam() : G(%d) : %f\n", class_no, G(class_no));
			# compare discriminant values with the current maximum and assign the appropriate class
			if G(class_no) > maxProb
				maxProb = G(class_no);
				C = class_no;
			endif
		endfor	
		
		#printf("detectSpam() : True Class : %d, Predicted Class : %d\n", rowMclass, C);
		CM(rowMclass,C)++;
	endfor
	
	printCM(CM);
endfunction

################# END OF FUNCTION DETECTSPAM() ###################
##################################################################



################## FUNCTION TO PRINT THE CONFUSION MATRIX #######################
################## ====================================== #######################


function printCM(CM)
	global no_classes
	
	printf("\n\n");
	printf("Confusion Matrix\n\n");
	printf(" ");
	for colno = [1:no_classes]
		printf("_______");
	endfor
	printf("\n");
	for rowno = [1:no_classes]
		for colno = [1:no_classes]
			printf("| %4d ",CM(rowno,colno));
		endfor
		printf("|\n");
		for colno = [1:no_classes]
			printf("|______");
		endfor
		printf("|\n");
	endfor
	printf("\n\n");
endfunction

################# END OF FUNCTION PRINTCM() ###################
###############################################################



################## FUNCTION TO PRINT THE EVALUATION PARAMETERS #######################
################## =========================================== #######################


function printStats(CM)
	global no_classes
	
	TP = zeros(1,no_classes);
	TN = zeros(1,no_classes);
	FP = zeros(1,no_classes);
	FN = zeros(1,no_classes);

	rowSums = zeros(1,no_classes);
	colSums = zeros(1,no_classes);
	matrixSum = 0;

	for rowno = [1:no_classes]
		for colno = [1:no_classes]
			rowSums(rowno) += CM(rowno,colno);
			colSums(colno) += CM(rowno,colno);
			matrixSum++;
		endfor
	endfor

	sumTP = 0;
	sumTPFP = 0;
	sumTPFN = 0;
	p_macro = 0;
	r_macro = 0;

	for class_no = [1:no_classes]
		TP(class_no) = CM(class_no,class_no);
		FN(class_no) = rowSums(class_no) - TP(class_no);
		FP(class_no) = colSums(class_no) - TP(class_no);
		TN(class_no) = matrixSum - (TP(class_no) + FN(class_no) + FP(class_no));
		sumTP += TP(class_no);
		sumTPFP += (TP(class_no) + FP(class_no));
		sumTPFN += (TP(class_no) + FN(class_no));
		p_macro += TP(class_no)/(TP(class_no) + FP(class_no));
		r_macro += TP(class_no)/(TP(class_no) + FN(class_no));
	endfor

	p_macro /= no_classes;
	r_macro /= no_classes;
	p_micro = sumTP/sumTPFP;
	r_micro = sumTP/sumTPFN;

	printf("\nMACRO PRECISION  : %f",p_macro);
	printf("\nMACRO RECALL  : %f",r_macro);
	printf("\nMICRO PRECISION  : %f",p_micro);
	printf("\nMICRO RECALL  : %f",r_micro);
	printf("\n\n");

endfunction

################# END OF FUNCTION PRINTSTATS() ###################
##################################################################



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        MAIN CODE        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# generate training and test ranges for n-cross validation
# ========================================================

CMG = zeros(no_classes);						# confusion matrix

for testCaseNo = [1 3 4]
	# put the corresponding fold as testCaseNo in the testData
	testData = D((testCaseNo - 1) * foldSize + 1:testCaseNo*foldSize,:);
	printf("%d : testData : %d %d\n", testCaseNo, rows(testData), columns(testData));
	# put rest of the folds into training data
	trainData = zeros(0,no_features);
	for rangeNo = [1:valFold]
		if rangeNo != testCaseNo
			trainData = [trainData; D((rangeNo - 1) * foldSize + 1:rangeNo*foldSize,:)];
		endif
	endfor
	printf("%d : trainData : %d %d\n", testCaseNo, rows(trainData), columns(trainData));
	printf("%d : Building Model\n", testCaseNo);
	LP = buildModel(trainData);
	printf("%d : Parameters : %d %d\n", testCaseNo, rows(LP), columns(LP));
	#disp(LP);
	printf("%d : Testing\n", testCaseNo);
	CMG = CMG .+ detectSpam(testData, LP);
	printf("%d : Testing complete\n\n", testCaseNo);
endfor

# Display the results
# ===================

printCM(CMG);
printStats(CMG);
