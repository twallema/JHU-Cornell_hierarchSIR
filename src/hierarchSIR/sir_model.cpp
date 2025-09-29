// dependencies
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/numeric/odeint.hpp>
#include <vector>
#include <cmath>
namespace py = pybind11;
using namespace boost::numeric::odeint;

int num_states = 10;

// SIR model
struct SIR {
    double rho_i, T_h;
    std::vector<double> beta, gamma, rho_h;
    std::vector<double> beta_modifiers;

    SIR(const std::vector<double>& beta, const std::vector<double>& gamma,
        double rho_i, const std::vector<double>& rho_h, double T_h,
        const std::vector<double>& beta_modifiers)
        : beta(beta), gamma(gamma), rho_i(rho_i), rho_h(rho_h), T_h(T_h), beta_modifiers(beta_modifiers) {}

    void operator()(const std::vector<double>& y, std::vector<double>& dydt, double t) {
        int num_strains = beta.size();
        int state_size = num_states * num_strains;  
        std::vector<double> T(num_strains, 0.0);

        // Compute total population for each strain
        for (int strain = 0; strain < num_strains; ++strain) {
            int idx = strain * num_states;
            T[strain] = y[idx] + y[idx + 1] + y[idx + 2];
        }

        // Get right modifier values
        int t_int = static_cast<int>(t);
        int index = t_int + 30;
        double modifier = (index >= 0 && index < beta_modifiers.size()) ? beta_modifiers[index] : 0.0;

        // Compute SIR model for every strain
        for (int strain = 0; strain < num_strains; ++strain) {
            int idx = strain * num_states;
            double beta_t = beta[strain] * (modifier+1);
            double S = y[idx], I = y[idx + 1], R = y[idx + 2], I_inc = y[idx + 3], H_inc_LCT0 = y[idx+4],  H_inc_LCT1 = y[idx+5], H_inc_LCT2 = y[idx+6], H_inc_LCT3 = y[idx+7], H_inc_LCT4 = y[idx+8], H_inc = y[idx+9];
            double lambda = beta_t * S * I / T[strain];

            dydt[idx] = -lambda;                                                    // dS/dt
            dydt[idx + 1] = lambda - gamma[strain] * I;                                  // dI/dt
            dydt[idx + 2] = gamma[strain] * I;                                           // dR/dt
            dydt[idx + 3] = rho_i * lambda - I_inc;                                 // dI_inc/dt
            dydt[idx + 4] = rho_h[strain] * lambda - (5/T_h) * H_inc_LCT0;          // dH_inc_LCT0/dt
            dydt[idx + 5] = (5/T_h) * H_inc_LCT0 - (5/T_h) * H_inc_LCT1;            // dH_inc_LCT1/dt
            dydt[idx + 6] = (5/T_h) * H_inc_LCT1 - (5/T_h) * H_inc_LCT2;            // dH_inc_LCT2/dt
            dydt[idx + 7] = (5/T_h) * H_inc_LCT2 - (5/T_h) * H_inc_LCT3;            // dH_inc_LCT3/dt
            dydt[idx + 8] = (5/T_h) * H_inc_LCT3 - (5/T_h) * H_inc_LCT4;            // dH_inc_LCT4/dt
            dydt[idx + 9] = (5/T_h) * H_inc_LCT4 - H_inc;                           // dH_inc/dt
        }
    }
};

// Gaussian smoothing function
std::vector<double> gaussian_smooth(const std::vector<double>& input, double sigma) {
    int size = static_cast<int>(sigma * 3);  // Kernel size (3 * sigma rule)
    std::vector<double> kernel(size * 2 + 1);
    std::vector<double> output(input.size(), 0.0);
    double sum = 0.0;

    // Compute Gaussian kernel
    for (int i = -size; i <= size; ++i) {
        kernel[i + size] = std::exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + size];
    }
    for (double& k : kernel) k /= sum;  // Normalize kernel

    // Convolve input with Gaussian kernel
    for (int i = 0; i < input.size(); ++i) {
        double smoothed = 0.0;
        for (int j = -size; j <= size; ++j) {
            int index = i + j;
            if (index >= 0 && index < input.size()) {
                smoothed += input[index] * kernel[j + size];
            }
        }
        output[i] = smoothed;
    }

    return output;
}

// Function to compute daily beta modifiers with padding and smoothing
std::vector<double> process_beta_modifiers(const std::vector<double>& delta_beta_temporal, 
                                           int modifier_length, double sigma) {
                                     
    // Step 1: Expand modifier vector to daily values using modifier length
    int full_length = modifier_length * static_cast<int>(delta_beta_temporal.size());
    std::vector<double> daily_beta(full_length, 0.0);
    for (int i = 0; i < delta_beta_temporal.size(); ++i) {
        for (int j = 0; j < modifier_length; ++j) {
            int idx = i * modifier_length + j;
            daily_beta[idx] = delta_beta_temporal[i];
        }
    }

    // Step 2: Pad with 30 extra days of `0.0`
    std::vector<double> padded_beta(30, 0.0);
    padded_beta.insert(padded_beta.end(), daily_beta.begin(), daily_beta.end());
    padded_beta.insert(padded_beta.end(), 30, 0.0);

    // Step 3: Apply Gaussian smoothing
    return gaussian_smooth(padded_beta, sigma);
}

// Interpolation function for output
std::vector<std::vector<double>> interpolate_results(const std::vector<std::vector<double>>& results, double t_start, double t_end) {
    std::vector<std::vector<double>> interpolated;
    int num_cols = results[0].size();
    size_t j = 0;
    for (double t = t_start; t <= t_end; ++t) {
        while (j + 1 < results.size() && results[j + 1][0] <= t) {
            ++j;
        }
        std::vector<double> interp_row(num_cols, 0.0);
        interp_row[0] = t;
        if (results[j][0] == t) {
            for (int col = 1; col < num_cols; ++col) {
                interp_row[col] = results[j][col];
            }
        } else if (j + 1 < results.size()) {
            double t1 = results[j][0], t2 = results[j + 1][0];
            for (int col = 1; col < num_cols; ++col) {
                double y1 = results[j][col], y2 = results[j + 1][col];
                interp_row[col] = y1 + (y2 - y1) * (t - t1) / (t2 - t1);
            }
        }
        interpolated.push_back(interp_row);
    }
    return interpolated;
}

// Function to integrate the SIR model
std::vector<std::vector<double>> solve(double t_start, double t_end,
                                        double atol, double rtol,
                                        std::vector<double> S0, std::vector<double> I0, std::vector<double> R0,
                                        std::vector<double> beta, std::vector<double> gamma, double rho_i, std::vector<double> rho_h, double T_h,
                                        const std::vector<double>& delta_beta_temporal, 
                                        int modifier_length, double sigma
                                        ) {
    double dt = 1;                  // initial guess for step size
    int num_strains = S0.size();    // determines number of strains
    std::vector<double> beta_modifiers = process_beta_modifiers(delta_beta_temporal, modifier_length, sigma);

    // Flatten initial conditions for integration
    std::vector<double> y;
    for (int strain = 0; strain < num_strains; ++strain) {
        y.push_back(S0[strain]);
        y.push_back(I0[strain]);
        y.push_back(R0[strain]);
        y.push_back(0.0);   // I_inc
        y.push_back(0.0);   // H_inc_LCT0
        y.push_back(0.0);   // H_inc_LCT1
        y.push_back(0.0);   // H_inc_LCT2
        y.push_back(0.0);   // H_inc_LCT3
        y.push_back(0.0);   // H_inc_LCT4
        y.push_back(0.0);   // H_inc
    }

    // Observe only S, I, R, I_inc and H_inc: omit linear chain trick
    std::vector<std::vector<double>> results;
    auto observer = [&](const std::vector<double>& y, double t) {
        std::vector<double> row = {t};
        // perform ommission
        for (int strain = 0; strain < num_strains; ++strain) {
            int idx = strain * num_states;  
            row.push_back(y[idx]);       // S
            row.push_back(y[idx + 1]);   // I
            row.push_back(y[idx + 2]);   // R
            row.push_back(y[idx + 3]);   // I_inc
            row.push_back(y[idx + 9]);   // H_inc (skip H_inc_LCT states)
        }
        results.push_back(row);
    };
    
    SIR sir_system(beta, gamma, rho_i, rho_h, T_h,  beta_modifiers);
    runge_kutta_dopri5<std::vector<double>> stepper;
    integrate_adaptive(make_controlled(atol, rtol, stepper), sir_system, y, t_start, t_end, dt, observer);
    return interpolate_results(results, t_start, t_end);
}

// Bind C++ module to Python
PYBIND11_MODULE(sir_model, m) {
    m.def("integrate", &solve, "Solve a strain-stratified SIR model",
          py::arg("t_start"), py::arg("t_end"),                                                         // solve between t_start and t_end
          py::arg("atol"), py::arg("rtol"),                                                             // solver accuracy        
          py::arg("S0"), py::arg("I0"), py::arg("R0"),                                                  // initial condition
          py::arg("beta"), py::arg("gamma"), py::arg("rho_i"), py::arg("rho_h"), py::arg("T_h"),        // SIR parameters
          py::arg("delta_beta_temporal"), py::arg("modifier_length"), py::arg("sigma")                  // modifier parameters
          );
}
