(ns pea-brain.core)

(defn relu [x]
  (if (neg? x) 0 x))

(defn relu' [x]
  (if (pos? x) 1 0))

(defn weights->activations [weights arguments]
  {:inputs arguments
   :output (->> arguments
                (map * (:inputs weights))
                (reduce + (:output weights))
                relu)})

(defn activate [[layer & net] arguments]
  (when layer
    (let [activated (map #(weights->activations % arguments) layer)]
      (cons activated (activate net (map :output activated))))))

(defn activations->gradients [activations gradient]
  (let [regularised-gradient (* gradient (relu' (:output activations)))]
    {:output regularised-gradient
     :inputs (->> (:inputs activations)
                  (map (partial * regularised-gradient)))}))

(defn grade-layer [layer gradients]
  (map activations->gradients layer gradients))

(defn input-gradients [net]
  (apply map + (map :inputs (first net))))

(defn back-propagate [[layer & net] gradients]
  (if net
    (let [remainder (back-propagate net gradients)]
      (cons (grade-layer layer (input-gradients remainder)) remainder))
    [(grade-layer layer gradients)]))

(defn neuron-by-neuron [f & nets]
  (apply map (partial map f) nets))

(defn value-by-value [f net1 net2]
  {:inputs (map f (:inputs net1) (:inputs net2))
   :output (f (:output net1) (:output net2))})

(defn move [weights gradients]
  (let [learning-rate 0.01
        adjust-with (fn [weight gradient] (+ weight (* learning-rate gradient)))]
    (value-by-value adjust-with weights gradients)))

(defn single-result [activations]
  (->> activations last first :output))

(defn grade [net inputs output]
  (let [activated (-> net (activate inputs))
        actual (single-result activated)
        improvement-gradient (- output actual)]
    (-> activated (back-propagate [improvement-gradient]))))

(defn train [net inputs expected]
  (let [graded (grade net inputs expected)]
    (neuron-by-neuron move net graded)))

(defn train-all [net [[input output] & cases]]
  (if input
    (train-all
      (train net input output)
      cases)
    net))

(defn train-iteratively [net n cases]
  (let [iterations (iterate #(train-all % cases) net)]
    (nth iterations n)))
