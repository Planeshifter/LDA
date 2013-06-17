#include <RcppArmadillo.h>
// [[Rcpp::depends("RcppArmadillo")]]

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>

using namespace Rcpp;
using namespace arma;
using namespace std;
using namespace boost::multiprecision;


class LDA {
public:
 int K; // K: number of topics
 int W; // W: size of Vocabulary
 int D; // D: number of documents
 vector< vector<int> >  w_num;
 vector< vector<int> >  z;
 vector<int> nd_sum;
 vector<int> nw_sum;
 NumericMatrix nd;
 NumericMatrix nw;
 NumericMatrix phi_avg;
 arma::mat n_wd;
 vector< vector < vector<int> > > z_list;
 vector< NumericMatrix > phi_list;
 vector< NumericMatrix > theta_list;
 NumericMatrix theta_avg;
 CharacterVector Vocabulary; // vector storing all (unique) words of vocabulary
 double alpha; // hyper-parameter for Dirichlet prior on theta
 double beta; //  hyper-parameter for Dirichlet prior on phi
 boost::mt19937 rng; // seed for random sampling

 List a;

LDA(Reference Obj);
void collapsedGibbs(int iter, int burnin, int thin);
void NichollsMH(int iter, int burnin, int thin);
NumericVector DocTopics(int d, int k);
NumericMatrix Topics(int k);
CharacterVector TopicTerms(int k, int no);
CharacterMatrix Terms(int k);
rowvec rDirichlet(double alpha, int length);
mat DrawFromProposal();
List getPhiList();
List getZList();
double PhiDensity2(NumericMatrix phi);
NumericMatrix PhiGradient(NumericMatrix phi, int z);

private:
vector< vector<int> > CreateIntMatrix(List input);
vector<cpp_dec_float_100> phiProd; 

NumericMatrix get_phis();
NumericMatrix get_thetas();
NumericMatrix MatrixToR(NumericMatrix input);
NumericMatrix avgMatrix(NumericMatrix A, NumericMatrix B, int weight);
NumericMatrix getTDM(int W, int D, List w_num);

cpp_dec_float_100 PhiDensity(arma::mat phi);
cpp_dec_float_100 ProposalDensity(arma::mat phi);

double rgamma_cpp(double alpha);



};

LDA::LDA(Reference Obj)
{

K = as<int>(Obj.field("K"));
W = as<int>(Obj.field("W"));
D = as<int>(Obj.field("D"));

nw = as<NumericMatrix> (Obj.field("nw"));
nd = as<NumericMatrix> (Obj.field("nd"));

alpha = Obj.field("alpha");
beta = Obj.field("beta");

Vocabulary = Obj.field("Vocabulary");

List temp_w = Obj.field("w_num");
List temp_z = Obj.field("z");
w_num = CreateIntMatrix(temp_w);

NumericMatrix tdm = getTDM(W, D, temp_w);
int i = tdm.nrow(), j = tdm.ncol();
arma::mat n_wd_pointer(tdm.begin(), i, j, false);
n_wd = n_wd_pointer;

z = CreateIntMatrix(temp_z);

nd_sum = as<vector<int> > (Obj.field("nd_sum"));
nw_sum = as<vector<int> > (Obj.field("nw_sum"));


a = List::create(Named("z")=z);
};

List LDA::getPhiList()
{
int iter = phi_list.size();
List ret(iter);

for (int i = 0; i<iter; i++)
  {
  ret[i] = phi_list[i];
	}
return ret;
}

List LDA::getZList()
{
int length = z_list.size();
List ret(length);

for (int i = 0; i<length; i++)
  {
  ret[i] = wrap(z_list[i]);
  }
return ret;
}

vector< vector<int> > LDA::CreateIntMatrix(List input)
    {

    int inputLength = input.size();
    vector< vector<int> > output;

    for(int i=0; i<inputLength; i++) {
              vector<int> test = as<vector<int> > (input[i]);
              output.push_back(test);
              }

    return output;

    }

NumericVector LDA::DocTopics(int d, int k)
  {
  vector<double> d_theta(K);
  NumericVector d_theta_R = theta_avg(d,_);
  d_theta = as<vector<double> > (d_theta_R);
  NumericVector ret_vector(k);

  for (int i=0;i<k;i++)
    {
    std::vector<double>::iterator result;
    result = std::max_element(d_theta.begin(),d_theta.end());
    int biggest_id = std::distance(d_theta.begin(), result);
    ret_vector[i] = biggest_id;
    d_theta[biggest_id] = 0;
    }

  return ret_vector;
  }

NumericMatrix LDA::Topics(int k)
  {
  NumericMatrix ret(D,k);
  for (int i = 0; i<D; i++)
    {
    NumericVector temp = DocTopics(i,k);
    ret(i,_) = temp;
    }
  ret = MatrixToR(ret);
  return ret;
  }

CharacterVector LDA::TopicTerms(int k, int no)
  {
  vector<double> k_phi(W);
  NumericVector k_phi_R = phi_avg(k,_);
  k_phi = as<vector<double> > (k_phi_R);
  NumericVector ret_vector(no);

  for (int i=0;i<no;i++)
    {
    std::vector<double>::iterator result;
    result = std::max_element(k_phi.begin(),k_phi.end());
    int biggest_id = std::distance(k_phi.begin(), result);
    ret_vector[i] = biggest_id;
    k_phi[biggest_id] = 0;
    }

  CharacterVector ret_char_vector(no);

  for (int i=0;i<no;i++)
    {
    ret_char_vector[i] = Vocabulary[ret_vector[i]];
    }

  return ret_char_vector;

  }

CharacterMatrix  LDA::Terms(int k)
  {
  CharacterMatrix ret(K,k);
  for (int i = 0; i < K; i++)
    {
    CharacterVector temp = TopicTerms(i,k);
    ret(i,_) =  temp;
    }
  return ret;
  }

NumericMatrix LDA::MatrixToR(NumericMatrix input)
  {
  int n = input.nrow(), k = input.ncol();
  NumericMatrix output(n,k);
  for (int i = 0; i<n; i++)
    {
    for (int j = 0; j<k; j++)
      {
      output(i,j) = input(i,j) + 1;
      }
    }
  return output;
  }

void LDA::collapsedGibbs(int iter, int burnin, int thin)
  {

  double Kd = (double) K;
  double Wd = (double) W;
  double W_Beta  = Wd * beta;
  double K_Alpha = Kd * alpha;

   for (int i = 0; i < iter; ++i)
          {
            for (int d = 0; d < D; ++d)
            {
              for (int w = 0; w < nd_sum[d]; ++w)
              {
              int word = w_num[d][w] - 1;
              int topic = z[d][w] - 1;

              nw(word,topic) -= 1;
              nd(d,topic) -= 1;
              nw_sum[topic] -= 1;
              nd_sum[d] -=  1;

              vector<double>  prob(K);

              for(int j=0; j<K; j++)
                {
                double nw_ij = nw(word,j);
                double nd_dj = nd(d,j);
                prob[j] = (nw_ij + beta) / (nw_sum[j] + W_Beta) *
                                (nd_dj + alpha) / (nd_sum[d] + K_Alpha);
                }

              for (int r = 1; r < K; ++r)
              {
              prob[r] = prob[r] + prob[r - 1];
              }

              double u  = prob[K-1] * rand() / double(RAND_MAX);

              int new_topic = 0; // set up new topic

              for (int nt = 0 ; nt < K; ++nt)
                {
                if (prob[nt] > u)
                  {
                  new_topic = nt;
                  break;
                  }
                }

               //  assign new z_i to counts
                 nw(word,new_topic) +=  1;
                 nd(d,new_topic) += 1;
                 nw_sum[new_topic] += 1;
                 nd_sum[d] += 1;

                 z[d][w] = new_topic + 1;

              }

            }



          if (i % thin == 0 && i > burnin)
            {
              z_list.push_back(z);

              NumericMatrix current_phi = get_phis();
              NumericMatrix current_theta = get_thetas();
              phi_list.push_back(current_phi);
              theta_list.push_back(current_theta);

              if(phi_list.size()==1) phi_avg = current_phi;
              else phi_avg =  avgMatrix(phi_avg, current_phi, phi_list.size());

              if(theta_list.size()==1) theta_avg = current_theta;
              else theta_avg =  avgMatrix(theta_avg, current_theta, theta_list.size());

            }

          }
  }

NumericMatrix LDA::avgMatrix(NumericMatrix A, NumericMatrix B, int weight)
  {
  int nrow = A.nrow();
  int ncol = A.ncol();
  NumericMatrix C(nrow,ncol);

  float wf = (float) weight;
  float propA = (wf-1) / wf;
  float propB = 1 / wf;

  for (int i=0; i<nrow;i++)
  {
    for (int j=0; j<ncol;j++)
    {
    C(i,j) =  propA * A(i,j) + propB * B(i,j);
    }
  }

  return C;
  }


NumericMatrix LDA::get_phis()
    {

      NumericMatrix phi(K,W);

       for (int k = 0; k < K; k++) {
         for (int w = 0; w < W; w++) {
           phi(k,w) = (nw(w,k) + beta) / (nw_sum[k] + W * beta);
         }
       }

      return phi;
    }

NumericMatrix LDA::get_thetas()
   {

    NumericMatrix theta(D,K);

     for (int d = 0; d<D; d++) {
       for (int k = 0; k<K; k++) {
         theta(d,k) = (nd(d,k) + alpha) / (nd_sum[d] + K * alpha);
       }
     }
   return theta;
   }

NumericMatrix LDA::getTDM(int W, int D, List w_num) {

                   NumericMatrix tdm(W,D);
                   for (int d=0; d<D; ++d)
                     for (int w=0; w<W; ++w)
                      {
                      int freq = 0;
                      vector<int> current_w = as<vector<int> > (w_num[d]);
                      int wlen = current_w.size();
                      for (int l=0; l<wlen; ++l)
                        {
                        if(current_w[l] == w + 1) freq += 1;
                        }

                       tdm(w,d) = freq;
                      }
                   return tdm;
                  }

// using R: (unfortunately too slow to call R and convert objects back)
//NumericMatrix LDA::DrawFromProposal()
//  {
//    Environment MCMCpack("package:MCMCpack");
//    Function rdirichlet = MCMCpack["rdirichlet"];
//    return rdirichlet(K,rep(beta,W));
//  }

double LDA::rgamma_cpp(double alpha)
  {
    boost::gamma_distribution<> dgamma(alpha);
    boost::variate_generator<boost::mt19937&,boost::gamma_distribution<> > ret_gamma( rng, dgamma);
    return ret_gamma();
  }

rowvec LDA::rDirichlet(double alpha, int length)
  {
  rowvec ret(length);
  for (int l = 0; l<length; l++)
    {
    ret[l] = rgamma_cpp(alpha);
    }
  ret = ret / sum(ret);
  return ret;
  }

mat LDA::DrawFromProposal()
    {
    mat phi(K,W);
    for (int k=0;k<K;k++)
      {
      phi.row(k) = rDirichlet(alpha,W);
      }
    return phi;
    }


void LDA::NichollsMH(int iter, int burnin, int thin)
  {

    mat phi_current = DrawFromProposal();
    for (int t=1;t<iter;t++)
    {

    // Metropolis-Hastings Algorithm:
    // 1. draw from proposal density:
    mat phi_new = DrawFromProposal();

    // 2. Calculate acceptance probability
    cpp_dec_float_100 pi_new = PhiDensity(phi_new);
    cpp_dec_float_100 pi_old = PhiDensity(phi_current);
    cpp_dec_float_100 q_new = ProposalDensity(phi_new);
    cpp_dec_float_100 q_old = ProposalDensity(phi_current);

    cpp_dec_float_100 acceptanceMH = (pi_new * q_old) / (pi_old * q_new);
    cpp_dec_float_100 alphaMH = std::min<boost::multiprecision::cpp_dec_float_100>(1,acceptanceMH);

    // draw U[0,1] random variable
    double u  = rand() / double(RAND_MAX);
    if (u<=alphaMH) phi_current = phi_new;
    else phi_current = phi_current;

    if (t % thin == 0 && t > burnin) {
     NumericMatrix phi_add = wrap(phi_current);
     phi_list.push_back(phi_add);
      if(phi_list.size()==1) phi_avg = phi_add;
              else phi_avg =  avgMatrix(phi_avg, phi_add, phi_list.size());
    };

    // Rcout << pi_new;
    // Rcout << pi_old;
    // Rcout << q_new;
    // Rcout << q_old;


    }

  }
  
  
NumericMatrix LDA::PhiGradient(NumericMatrix phi, int z)
  {
    arma::mat phi2 = as<arma::mat>(phi);
    arma::mat logPhi = log(phi2);
    double logPhiProd[K];
    double PhiProd[K];
    double dSum = 0; 
    arma::mat gradient(K,W);
    
    for (int z=0;z<K;z++)
    {
      for(int w=0;w<W;w++)
      {
        
      for (int d = 0; d<D;d++)
      {  
        double sumLik = 0;
        double nwd = n_wd(w,d);
        arma::colvec nd = n_wd.col(d);
  
        for (int k=0; k<K; k++)
         {
         arma::rowvec logPhi_k = logPhi.row(k);
         logPhiProd[k] = dot(logPhi_k,nd);
         PhiProd[k] = exp(logPhiProd[k]);        
         sumLik += PhiProd[k];
         }
      
        dSum +=  (nwd / phi2(z,w)) * PhiProd[z] / sumLik;       
      }
    
      gradient(z,w) = dSum + (beta - 1) / phi2(z,w);
          
      }
    }
       
    return wrap(gradient);   
  }  
  
  

cpp_dec_float_100 LDA::ProposalDensity(arma::mat phi)
  {
    cpp_dec_float_100 logBetaFun = K*(W*lgamma(beta)-lgamma(W*beta));
    arma::mat logPhi = log(phi);
    arma::mat temp = logPhi * (beta-1);
    cpp_dec_float_100 logPhiSum = accu(temp);

    cpp_dec_float_100 logDensity = logPhiSum - logBetaFun;
    Rcout << exp(logDensity) << " - ";
    return exp(logDensity);
  }

cpp_dec_float_100 LDA::PhiDensity(arma::mat phi)
  {
  arma::mat logPhi = log(phi);
  cpp_dec_float_100 logLikelihood_vec[D];
  cpp_dec_float_100 sumLik_vec[K];
  cpp_dec_float_100 logLikelihood = 0;

   for (int d=0; d<D; d++)
     {
     cpp_dec_float_100 sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       sumLik_vec[k] = dot(logPhi_k,nd);
       sumLik += sumLik_vec[k];
       }
     cpp_dec_float_100 b = ArrayMax(sumLik_vec);
     logLikelihood_vec[d] = exp(sumLik + b);
     Rcout << logLikelihood_vec[d] << " - ";
     Rcout << "b:" << b << " - ";
     logLikelihood += b + log(logLikelihood_vec[d]);
     }
      logLikelihood += D * log(alpha);
      logLikelihood -= D * log(K*alpha);
      //Rcout << "logLikelihood: " << logLikelihood;

      cpp_dec_float_100 logBetaFun = K*(lgamma(W*beta)-W*lgamma(beta));
      //Rcout << "logBetaFun: " << logBetaFun;

      cpp_dec_float_100 logPhiSum = 0;

      arma::mat temp = logPhi * (beta-1);
      logPhiSum = accu(temp);

      //Rcout << "LogPhiSum: " << logPhiSum;

    cpp_dec_float_100 logProb = logLikelihood + logBetaFun + logPhiSum;
    cpp_dec_float_100 Prob = exp(logProb);
    return Prob;
   }

double LDA::PhiDensity2(NumericMatrix phi)
  {
  arma::mat phi2 = as<arma::mat>(phi);
  arma::mat logPhi = log(phi2);
  double logLikelihood_vec[D];
  double logLikelihood = 0;

   for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       double inProd_k = 0;

       for (int w=0; w<W; w++)
  	   {
    	   inProd_k += logPhi_k[w] * nd[w];
		   }
       // Rcout << inProd_k;
       double sumLik_k = exp(inProd_k) * alpha;
       sumLik += sumLik_k;
       }
     logLikelihood_vec[d] = log(sumLik);
     logLikelihood += logLikelihood_vec[d];
     }

      logLikelihood -= D * log(K*alpha);
      //Rcout << "logLikelihood: " << logLikelihood;

      double logBetaFun = K*(lgamma(W*beta)-W*lgamma(beta));
      //Rcout << "logBetaFun: " << logBetaFun;

      double logPhiSum = 0;

      arma::mat temp = logPhi * (beta-1);
      logPhiSum = accu(temp);

      //Rcout << "LogPhiSum: " << logPhiSum;

    double logProb = logLikelihood + logBetaFun + logPhiSum;
    double Prob = exp(logProb);
    return Prob;
   }


cpp_dec_float_100 ArrayMax(cpp_dec_float_100 array[])
{
     int length = array.size( );  // establish size of array
     cpp_dec_float_100 max = array[0];       // start with max = first element

     for(int i = 1; i<length; i++)
     {
          if(array[i] > max)
                max = array[i];
     }
     return max;                // return highest value in array
}



RCPP_MODULE(LDA_module) {
class_<LDA>( "LDA" )
.constructor<Reference>()
//.field( "w_num", &LDA::w_num)
.field( "a", &LDA::a)
.field( "nd_sum", &LDA::nd_sum)
.field("nd",&LDA::nd)
.field( "nw_sum", &LDA::nw_sum)
.field("nw",&LDA::nw)
.field("K", &LDA::K)
.field("D",&LDA::D)
.field("phi_avg",&LDA::phi_avg)
.field("theta_avg",&LDA::theta_avg)
.method("collapsedGibbs",&LDA::collapsedGibbs)
.method("Topics",&LDA::Topics)
.method("Terms",&LDA::Terms)
.method("NichollsMH",&LDA::NichollsMH)
.method("DrawFromProposal",&LDA::DrawFromProposal)
.method("getPhiList",&LDA::getPhiList)
.method("getZList",&LDA::getZList)
.method("PhiGradient",&LDA::PhiGradient)
;
}
