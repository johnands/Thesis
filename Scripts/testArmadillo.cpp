    // Test if built-in element-wise product is faster than double loop, which it is 
    arma::mat A(1, 500, arma::fill::randu);
    arma::mat B(1, 500, arma::fill::randu);

    clock_t start, finish;
    start = clock();
    for (int i=0; i < 100000000; i++) {
        arma::mat product = A % B;
    }
    finish = clock();
    cout << "Time elapsed: " << ((finish-start)/CLOCKS_PER_SEC) << endl;
    
    start = clock();
    for (int i=0; i < int(1e8); i++) {
        arma::mat product(1,500);
        for (int j=0; j < 500; j++) {
            product(0,j) = A(0,j) * B(0,j);
        }
    }
    finish = clock();
    cout << "Time elapsed: " << ((finish-start)/CLOCKS_PER_SEC) << endl;
    

	// also test if it is faster to do product of two row vectors than two (1,N)-matrices
	// this is not the case
    arma::rowvec C(500, arma::fill::randu);
    arma::rowvec D(500, arma::fill::randu);

    start = clock();
    for (int i=0; i < 100000000; i++) {
        arma::rowvec product = C % D;
    }
    finish = clock();
    cout << "Time elapsed: " << ((finish-start)/CLOCKS_PER_SEC) << endl;
