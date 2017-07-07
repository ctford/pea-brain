(ns pea-brain.core-test
  (:require [clojure.test :refer :all]
            [pea-brain.core :refer :all]))

(defn round [x]
  (-> x (+ 0.5) int))

(def xor
  [[{:inputs [1.0 -1.0] :output 0.0} {:inputs [-1.0 1.0] :output 0.0}]
   [{:inputs [1.0 1.0]  :output 0.0}]])

(def half-xor
  [[{:inputs [1/2 -1/2] :output 0} {:inputs [-1/2 1/2] :output 0}]
   [{:inputs [1/2 1/2]  :output 0}]])

(def conjunction
  [[{:inputs [1.0 1.0] :output 0.0} {:inputs [1.0 1.0] :output 0.0}]
   [{:inputs [1.0 1.0] :output -3.0}]])

(def weakened-conjunction
  [[{:inputs [1 1] :output 0} {:inputs [1 1] :output 0}]
   [{:inputs [1 1] :output 0}]])

(deftest logic
  (testing "activation"
    (is (= 0.0 (single-result (activate xor [0 0]))))
    (is (= 1.0 (single-result (activate xor [1 0]))))
    (is (= 1.0 (single-result (activate xor [0 1]))))
    (is (= 0.0 (single-result (activate xor [1 1]))))
    (is (= 0 (single-result (activate conjunction [0 0]))))
    (is (= 0 (single-result (activate conjunction [1 0]))))
    (is (= 0 (single-result (activate conjunction [0 1]))))
    (is (= 1.0 (single-result (activate conjunction [1 1])))))

  (testing "grading"
    (is (= [[{:inputs [3/4 0] :output 3/4} {:inputs [0 0] :output 0}]
            [{:inputs [3/4 0] :output 3/2}]]
           (back-propagate (activate half-xor [1 0]) [(* 2 3/4)])))

    (is (= [[{:inputs [0 0] :output 0} {:inputs [0 3/4] :output 3/4}]
            [{:inputs [0 3/4] :output 3/2}]]
           (back-propagate (activate half-xor [0 1]) [(* 2 3/4)])))

    (is (= [[{:inputs [-12 -12] :output -12} {:inputs [-12 -12] :output -12}]
            [{:inputs [-12 -12] :output -6}]]
           (back-propagate (activate weakened-conjunction [1 1]) [(* 2 -3)])))

    (is (= [[{:inputs [0 -2] :output -2} {:inputs [0 -2] :output -2}]
            [{:inputs [-2 -2] :output -2}]]
           (back-propagate (activate weakened-conjunction [0 1]) [(* 2 -1)]))))

  (testing "training"
    (is (= conjunction
           (train-iteratively conjunction 1 [[[1 1] 1]])))

    (is (= [[{:inputs [0.94 0.94] :output -0.06} {:inputs [0.94 0.94] :output -0.06}]
            [{:inputs [0.94 0.94] :output -0.03}]]
           (train-iteratively weakened-conjunction 1 [[[1 1] 1]])))

    (is (= [0 0 0 1]
           (let [fixed-conjunction (-> weakened-conjunction
                                       (train-iteratively 10 [[[0 0] 0]
                                                              [[0 1] 0]
                                                              [[1 0] 0]
                                                              [[1 1] 1]]))]
             [(-> fixed-conjunction (activate [0 0]) single-result round)
              (-> fixed-conjunction (activate [0 1]) single-result round)
              (-> fixed-conjunction (activate [1 0]) single-result round)
              (-> fixed-conjunction (activate [1 1]) single-result round)])))

    (is (= [[{:inputs [1.0 0.98] :output -0.02} {:inputs [1.0 0.98] :output -0.02}]
            [{:inputs [0.98 0.98] :output -0.02}]]
           (train-iteratively weakened-conjunction 1 [[[0 1] 0]])))

    (is (= xor
           (train-iteratively xor 10 [[[1 0] 1]])))

    (is (= [[{:inputs [0.50375 -0.5] :output 0.00375} {:inputs [-0.5 0.5] :output 0.0}]
            [{:inputs [0.50375 0.5] :output 0.0075}]]
           (train-iteratively half-xor 1 [[[1 0] 1]])))

    (is (= [0 1 1 0]
           (let [fixed-xor (-> half-xor
                               (train-iteratively 1000 [[[0 0] 0]
                                                        [[0 1] 1]
                                                        [[1 0] 1]
                                                        [[1 1] 0]]))]
             [(-> fixed-xor (activate [0 0]) single-result round)
              (-> fixed-xor (activate [0 1]) single-result round)
              (-> fixed-xor (activate [1 0]) single-result round)
              (-> fixed-xor (activate [1 1]) single-result round)])))))
