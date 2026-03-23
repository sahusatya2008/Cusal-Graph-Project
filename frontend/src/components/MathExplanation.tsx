import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronRight } from 'lucide-react';

export default function MathExplanation() {
  const [expandedSection, setExpandedSection] = useState<string | null>('do-calculus');

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="card-header flex items-center">
          <BookOpen className="mr-2" size={20} />
          Mathematical Foundations
        </h2>
        <p className="text-gray-600 mb-4">
          Understanding the mathematical theory behind causal inference and policy optimization.
        </p>

        <div className="space-y-4">
          {/* do-Calculus Section */}
          <div className="border rounded-lg">
            <button
              onClick={() => toggleSection('do-calculus')}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
            >
              <span className="font-semibold">do-Calculus</span>
              {expandedSection === 'do-calculus' ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
            </button>
            {expandedSection === 'do-calculus' && (
              <div className="p-4 pt-0 space-y-4">
                <p className="text-gray-600">
                  The <strong>do-calculus</strong> provides rules for transforming interventional distributions 
                  into observational ones, enabling causal inference from non-experimental data.
                </p>
                
                <div className="p-4 bg-indigo-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Rule 1: Insertion/Deletion of Observations</h4>
                  <p className="font-mono text-sm mb-2">P(Y | do(X), Z, W) = P(Y | do(X), W)</p>
                  <p className="text-sm text-gray-600">If Y is conditionally independent of Z given W.</p>
                </div>

                <div className="p-4 bg-green-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Rule 2: Action/Observation Exchange</h4>
                  <p className="font-mono text-sm mb-2">P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)</p>
                  <p className="text-sm text-gray-600">If Y is conditionally independent of Z given W.</p>
                </div>

                <div className="p-4 bg-purple-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Rule 3: Insertion/Deletion of Actions</h4>
                  <p className="font-mono text-sm mb-2">P(Y | do(X), do(Z), W) = P(Y | do(X), W)</p>
                  <p className="text-sm text-gray-600">If all back-door paths from Z to Y are blocked.</p>
                </div>
              </div>
            )}
          </div>

          {/* Truncated Factorization Section */}
          <div className="border rounded-lg">
            <button
              onClick={() => toggleSection('truncated')}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
            >
              <span className="font-semibold">Truncated Factorization</span>
              {expandedSection === 'truncated' ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
            </button>
            {expandedSection === 'truncated' && (
              <div className="p-4 pt-0 space-y-4">
                <p className="text-gray-600">
                  The <strong>truncated factorization formula</strong> computes interventional distributions 
                  by removing factors corresponding to intervened variables.
                </p>
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Formula</h4>
                  <p className="font-mono text-sm mb-2">P(Y | do(X=x)) = ∑ᵥ ∏ᵢ P(Vᵢ | PAᵢ) |ₓ</p>
                  <p className="text-sm text-gray-600">
                    Product over all variables except intervened X, PAᵢ = parents of Vᵢ.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Back-Door Criterion Section */}
          <div className="border rounded-lg">
            <button
              onClick={() => toggleSection('backdoor')}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
            >
              <span className="font-semibold">Back-Door Criterion</span>
              {expandedSection === 'backdoor' ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
            </button>
            {expandedSection === 'backdoor' && (
              <div className="p-4 pt-0 space-y-4">
                <p className="text-gray-600">
                  The <strong>back-door criterion</strong> identifies confounders for causal effect estimation.
                </p>
                
                <div className="p-4 bg-yellow-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Conditions</h4>
                  <ol className="list-decimal list-inside text-sm space-y-1">
                    <li>No node in Z is a descendant of X</li>
                    <li>Z blocks every back-door path from X to Y</li>
                  </ol>
                </div>

                <div className="p-4 bg-green-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Adjustment Formula</h4>
                  <p className="font-mono text-sm">P(Y | do(X=x)) = ∑ᵤ P(Y | X=x, Z=z) · P(Z=z)</p>
                </div>
              </div>
            )}
          </div>

          {/* Counterfactuals Section */}
          <div className="border rounded-lg">
            <button
              onClick={() => toggleSection('counterfactual')}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
            >
              <span className="font-semibold">Counterfactual Reasoning</span>
              {expandedSection === 'counterfactual' ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
            </button>
            {expandedSection === 'counterfactual' && (
              <div className="p-4 pt-0 space-y-4">
                <p className="text-gray-600">
                  <strong>Counterfactuals</strong> answer "What would have happened if..." questions.
                </p>
                
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Three-Step Process</h4>
                  <ol className="list-decimal list-inside text-sm space-y-2">
                    <li><strong>Abduction:</strong> Infer exogenous variables from evidence</li>
                    <li><strong>Action:</strong> Modify the model with intervention</li>
                    <li><strong>Prediction:</strong> Compute the counterfactual outcome</li>
                  </ol>
                </div>

                <div className="p-4 bg-purple-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Notation</h4>
                  <p className="font-mono text-sm">Yₓ(u) = Y in world where X=x, given unit U=u</p>
                </div>
              </div>
            )}
          </div>

          {/* Structural Equations Section */}
          <div className="border rounded-lg">
            <button
              onClick={() => toggleSection('structural')}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50"
            >
              <span className="font-semibold">Structural Causal Models</span>
              {expandedSection === 'structural' ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
            </button>
            {expandedSection === 'structural' && (
              <div className="p-4 pt-0 space-y-4">
                <p className="text-gray-600">
                  <strong>Structural Causal Models (SCMs)</strong> formalize causal relationships.
                </p>
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Structural Equation</h4>
                  <p className="font-mono text-sm mb-2">Xᵢ = fᵢ(PAᵢ, Uᵢ)</p>
                  <p className="text-sm text-gray-600">
                    Each variable is a function of its parents and exogenous noise.
                  </p>
                </div>

                <div className="p-4 bg-indigo-50 rounded-lg">
                  <h4 className="font-semibold mb-2">Example: Linear SCM</h4>
                  <div className="font-mono text-sm space-y-1">
                    <p>X = Uₓ</p>
                    <p>Y = βX + Uᵧ</p>
                    <p>Z = γY + Uᵤ</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="card-header">Key References</h3>
        <ul className="space-y-2 text-sm">
          <li className="p-2 bg-gray-50 rounded">
            <strong>Pearl, J. (2009).</strong> Causality: Models, Reasoning, and Inference. Cambridge University Press.
          </li>
          <li className="p-2 bg-gray-50 rounded">
            <strong>Pearl, J. & Mackenzie, D. (2018).</strong> The Book of Why. Basic Books.
          </li>
          <li className="p-2 bg-gray-50 rounded">
            <strong>Peters, J., Janzing, D., & Schölkopf, B. (2017).</strong> Elements of Causal Inference. MIT Press.
          </li>
        </ul>
      </div>
    </div>
  );
}