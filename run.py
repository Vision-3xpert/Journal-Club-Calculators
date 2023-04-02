from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from io import BytesIO
import base64
import math



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/project1', methods=['GET', 'POST'])
def project1():
    if request.method == 'POST':
        # Extract user inputs from the form
        n_trials_1 = float(request.form['n_trials_1'])
        n_successes_1 = float(request.form['n_successes_1'])
        n_trials_2 = float(request.form['n_trials_2'])
        n_successes_2 = float(request.form['n_successes_2'])
        alpha1, beta1 = n_successes_1, n_trials_1 - n_successes_1
        alpha2, beta2 = n_successes_2, n_trials_2 - n_successes_2

        # Approximation des lois bêta avec des lois normales
        mu1, sigma1 = alpha1 / (alpha1 + beta1), np.sqrt(alpha1 * beta1 / ((alpha1 + beta1)**2 * (alpha1 + beta1 + 1)))
        mu2, sigma2 = alpha2 / (alpha2 + beta2), np.sqrt(alpha2 * beta2 / ((alpha2 + beta2)**2 * (alpha2 + beta2 + 1)))
        # Create a function to find the intersection
        def find_intersection(mu1, sigma1, mu2, sigma2):
            a = 1 / (2 * sigma1**2) - 1 / (2 * sigma2**2)
            b = mu2 / (sigma2**2) - mu1 / (sigma1**2)
            c = mu1**2 / (2 * sigma1**2) - mu2**2 / (2 * sigma2**2) - np.log((sigma2 * np.sqrt(2 * np.pi)) / (sigma1 * np.sqrt(2 * np.pi)))
            return np.roots([a, b, c])


        intersections = find_intersection(mu1, sigma1, mu2, sigma2)
        highest_intersection = intersections[np.argmax(np.array([norm.pdf(x, mu1, sigma1) for x in intersections]))]
        lower_limit = min(mu1 - 4*sigma1, mu2 - 4*sigma2)
        upper_limit = max(mu1 + 4*sigma1, mu2 + 4*sigma2)
        from scipy.integrate import trapz


        # Générer les valeurs de x en fonction des limites
        x = np.linspace(lower_limit, upper_limit, 1000)
        y1 = norm.pdf(x, mu1, sigma1)
        y2 = norm.pdf(x, mu2, sigma2)
        # Calculer les aires sous les courbes
        area_2 = trapz(y2, x, dx=2/1000)
        common_area = trapz(np.minimum(y1, y2), x, dx=2/1000)
        area_diff_2 = round((area_2 - common_area)*100,2)
        plt.plot(x, y1, label='Médicament 1')
        plt.plot(x, y2, label='Médicament 2')

        plt.axvline(highest_intersection, color='red', linestyle='dashed', label=f'Intersection')

        common_area = np.minimum(y1, y2)
        plt.fill_between(x, common_area, color='purple', alpha=0.3)

        plt.legend()
        plt.xlabel("Efficacité")
        plt.ylabel("Densité de probabilité")
        plt.title("Comparaison de l'efficacité des médicaments")

        # Save plot to a bytes buffer
        # Save plot to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Close the current figure
        plt.close()


        # Encode the bytes buffer to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return render_template('project1.html', plot=img_base64,area_diff=area_diff_2)
    return render_template('project1_input.html')

@app.route('/project2', methods=['GET', 'POST'])
def project2():
    plot_base64 = None
    if request.method == 'POST':
        x1 = int(request.form['x1'])
        x2 = int(request.form['x2'])
        n1 = int(request.form['n1'])
        n2 = int(request.form['n2'])
        def calculate_rr(x1, n1, x2, n2):
            pro1 = x1 / n1
            pro2 = x2 / n2
            return pro1 / pro2

        def calculate_confidence_interval(RR, x1, n1, x2, n2):
            inside1 = ((n1 - x1) / x1) / n1
            inside2 = ((n2 - x2) / x2) / n2
            insidebig = inside1 + inside2

            sqrt = math.sqrt(insidebig)
            KN = sqrt * 1.96

            logRR = np.log(RR)
            logLL = logRR - KN
            logUL = logRR + KN

            LL = round(np.exp(logLL), 2)
            UL = round(np.exp(logUL), 2)
            
            return LL, UL

        def plot_results(Y_list, LL_list, UL_list, RR_list):
            plt.scatter(Y_list, UL_list, marker="_", color="b")
            plt.scatter(Y_list, LL_list, marker="_", color="b")
            plt.scatter(Y_list, RR_list, marker="o", color="b")
            plt.xlabel("Number of iterations = Fragility Index")
            plt.ylabel("Risk Ratio")

            plt.grid()
            plt.axhline(y=1.0, color="r", linestyle="dashed", linewidth=5)
            plt.vlines(x=Y_list, ymin=LL_list, ymax=UL_list, linewidth=2)

            # Save plot to a buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Clear the current plot before returning the buffer
            plt.clf()

            return buf


        def save_plot_to_base64(buf):
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return plot_base64

        def main(x1, x2, n1, n2):
            x3, x4 = n1 - x1, n2 - x2
            UL_list, LL_list, RR_list, Y_list = [], [], [], []
            y = 0
            RR = calculate_rr(x1, n1, x2, n2)
            LL, UL = calculate_confidence_interval(RR, x1, n1, x2, n2)
            UL_list.append(UL)
            LL_list.append(LL)
            RR_list.append(RR)
            Y_list.append(y)
            if RR < 1:
                for i in range(0, 1000):
                    x1 += 1
                    x2 -= 1
                    x3 = (n1 - x1) - 1
                    RR = calculate_rr(x1, n1, x2, n2)
                    LL, UL = calculate_confidence_interval(RR, x1, n1, x2, n2)


                    UL_list.append(UL)
                    LL_list.append(LL)
                    RR_list.append(RR)
                    y += 1
                    Y_list.append(y)
    
                    if UL > 1.0:
                        img = plot_results(Y_list, LL_list, UL_list, RR_list)
                        return img

            else:
                    for i in range(0, 1000):
                        x2 += 1
                        x1 -= 1
                        x3 = (n1 - x1) - 1

                        RR = calculate_rr(x1, n1, x2, n2)
                        LL, UL = calculate_confidence_interval(RR, x1, n1, x2, n2)

                        UL_list.append(UL)
                        LL_list.append(LL)
                        RR_list.append(RR)
                        y += 1
                        Y_list.append(y)


                        if LL < 1:
                            img = plot_results(Y_list, LL_list, UL_list, RR_list)
                            return img

        
        img = main(x1, x2, n1, n2)
        plot_base64 = save_plot_to_base64(img)

    return render_template('project2.html', plot=plot_base64)

if __name__ == "__main__":
    app.run(debug=True)
