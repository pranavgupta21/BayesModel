# naive Bayes classifier on UCI spambase data set #
# assumption : the data is correct and preprocessed accurately without any anomalies #

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    GLOBAL VARIABLES     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# fetch the data set into a matrix
global D = csvread('synthetic-data');
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


################## FUNCTION TO BUILD THE NAIVE BAYES MODEL #######################
################## ======================================= #######################


function LP = buildModel(M)
	global D valFold foldSize no_features no_classes no_records;

	classSizes = zeros(1, no_classes);
	P = zeros(1, no_classes);
	likelihood_counts = ones((no_features-1), no_classes);
	likelihoods_prob = zeros((no_features-1), no_classes);

	for rowno = [1:rows(M)]
		rowM = M(rowno,:);
		class_no = rowM(no_features);
		classSizes(class_no)++;
		for featno = [1:(no_features - 1)]
			if rowM(featno) > 0
				likelihood_counts(featno, class_no)++;
			endif
		endfor
	endfor

	#printf("buildModel() : likelihood_counts : \n");
	#disp(likelihood_counts);
	
	#printf("buildModel() : classSizes : \n");
	#disp(classSizes);

	# Calculating Priors
	# ==================
	
	for class_no = [1:no_classes]
		P(class_no) = classSizes(class_no)/rows(M);
	endfor

	#printf("buildModel() : priors : \n");
	#disp(P);

	# Calculating Likelihoods
	# =======================

	for featno = [1:(no_features-1)]
		for class_no = [1:no_classes]
			#printf("[%d %d] : likelihood_count : %d\tDenominator : %d\n", featno, class_no, likelihood_counts(featno, class_no), (classSizes(class_no) + no_features - 1)); 
			likelihoods_prob(featno,class_no) = likelihood_counts(featno,class_no)/(classSizes(class_no) + no_features - 1);
		endfor
	endfor
	#printf("buildModel() : likelihoods : \n");
	#disp(likelihoods_prob);
	LP = [likelihoods_prob;P];	    # store the likelihoods (d*d matrix) and the priors(no_classes*1 matrix)
endfunction

################# END OF FUNCTION BUILDMODEL() ###################
##################################################################



################## FUNCTION TO CLASSIFY THE TEST DATA #######################
################## ================================== #######################


function CM = detectSpam (M,L,P)
	global D valFold foldSize no_features no_classes no_records;
	CM = zeros(no_classes,no_classes);
	
	for rowno = [1:rows(M)]
		rowM = M(rowno,:);

		# calculate the discriminant functions for all classes
		G = zeros(no_classes);			# G holds the discriminant function values for all classes

		# add log-likelihoods of all features (assuming that the features are independent)
		for class_no = [1:no_classes]
			for colno = [1:(no_features - 1)]
				if rowM(colno) == 0
					G(class_no) += log((1 - L(colno,class_no)));		# 1 - L() gives the likelihood of feature = 0
				else
					G(class_no) += log(L(colno,class_no));			# Likelihood matrix stores the likelihoods of feature = 1
				endif				
			endfor
		endfor
	
		# add the log-priors to calculate final probabilities and determine class
		G(1) += log(P(1));
		C = 1;
		maxProb = G(1);
		for class_no = [2:no_classes]
			G(class_no) += log(P(class_no));
			#printf("G(%d) : %f\n", class_no, G(class_no));
			if G(class_no) > maxProb
				maxProb = G(class_no);
				C = class_no;
			endif
		endfor
		
		#printf("True Class : %d, Chosen class : %d\n", rowM(no_features), C);
		# fill confusion matrix
		CM(rowM(no_features),C)++;
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

CMG = zeros(no_classes,no_classes);						# confusion matrix

for testCaseNo = [1:valFold]
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
	printf("%d : Model build\n", testCaseNo);
	L = LP(1:(no_features - 1),:);
	P = LP(no_features,:);
	printf("%d : Parameters : %d %d\n", testCaseNo, rows(LP), columns(LP));
	#disp(LP);
	printf("%d : Testing\n", testCaseNo);
	CMG = CMG .+ detectSpam(testData, L, P);
	printf("%d : Testing complete\n\n", testCaseNo);
endfor


# Display the results
# ===================

printCM(CMG);
printStats(CMG);
