#include <RcppArmadillo.h>
// [[Rcpp::depends("RcppArmadillo")]]

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/math/distributions.hpp>

#include <numeric>
#include <algorithm>
#include <map>
#include <string>
#include <iostream>

using namespace Rcpp;
using namespace arma;
using namespace std;
using namespace boost::multiprecision;
using namespace boost::math;


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
 vector<double> PhiProd_vec;
 List a;

LDA(Reference Obj);
void collapsedGibbs(int iter, int burnin, int thin);
void NichollsMH(int iter, int burnin, int thin);
NumericVector DocTopics(int d, int k);
NumericMatrix Topics(int k);
CharacterVector TopicTerms(int k, int no);
CharacterMatrix Terms(int k);
arma::rowvec rDirichlet(arma::rowvec param, int length);
arma::rowvec rDirichlet2(arma::rowvec param, int length);
arma::mat DrawFromProposal(arma::mat phi_current);
arma::mat DrawFromProposalInit();
List getPhiList();
List getZList();
double PhiDensity2(NumericMatrix phi);
NumericMatrix PhiGradient(NumericMatrix phi);
double rgamma_cpp(double alpha);
double rbeta_cpp(double shape1, double shape2);
double LogPhiProd(arma::mat phi); 
vector<double> LogPhiProd_vec(arma::mat phi);

private:
vector< vector<int> > CreateIntMatrix(List input);


NumericMatrix get_phis();
NumericMatrix get_thetas();
NumericMatrix MatrixToR(NumericMatrix input);
NumericMatrix avgMatrix(NumericMatrix A, NumericMatrix B, int weight);
NumericMatrix getTDM(int W, int D, List w_num);

double PhiDensity(arma::mat phi);
double ProposalDensity(arma::mat phi);
double ArrayMax(double array[], int numElements);
double ArrayMin(double array[], int numElements);



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

CharacterMatrix LDA::Terms(int k)
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


double LDA::rbeta_cpp(double shape1, double shape2)
  {
    double u  = rand() / double(RAND_MAX);
    beta_distribution<> beta_dist(shape1, shape2);
    return quantile(beta_dist, u);  
  }

double LDA::rgamma_cpp(double alpha)
  {
    boost::gamma_distribution<> dgamma(alpha);
    boost::variate_generator<boost::mt19937&,boost::gamma_distribution<> > ret_gamma( rng, dgamma);
    return ret_gamma();
  }

arma::rowvec LDA::rDirichlet(arma::rowvec param, int length)
  {
  rowvec ret(length);
  for (int l = 0; l<length; l++)
    {
    double beta = param[l];
    ret[l] = rgamma_cpp(beta);
    }
  ret = ret / sum(ret);
  return ret;
  }
  
arma::rowvec LDA::rDirichlet2(arma::rowvec param, int length)
  {
  vector<double> ret;
  param *= 10000;
  vector<double> param_vec = conv_to<vector<double> >::from(param);
  Rcout << param_vec[0] << "-";
  int len = length - 1;
  
  double paramSum = std::accumulate(param_vec.begin()+1,param_vec.end(),(double)0);
  Rcout << paramSum;
  ret.push_back(rbeta_cpp(param_vec[0], paramSum));
  for (int i=1; i<len;i++)
    {
    double paramSum = std::accumulate(param_vec.begin()+i+1,param_vec.end(),(double)0); 
    double phi = rbeta_cpp(param_vec[i], paramSum);
    double sumRet = std::accumulate(ret.begin(),ret.end(),(double)0);  
    ret.push_back((1-sumRet) * phi);
    }   
  double sumRet = std::accumulate(ret.begin(),ret.end(),(double)0); 
  ret.push_back(1-sumRet);
  return ret;
  }  
  

mat LDA::DrawFromProposal(arma::mat phi_current)
    {
    arma::mat phi_sampled(K,W);
    for (int k=0;k<K;k++)
      {
      arma::rowvec phi_current_row = phi_current.row(k);
      arma::rowvec new_row = rDirichlet2(phi_current_row, W);
      // Rcout << new_row;
      phi_sampled.row(k) = new_row;
      }
    return phi_sampled;
    }
    
mat LDA::DrawFromProposalInit()
    {
    arma::mat phi(K,W);
    arma::rowvec beta_vec(W);
    
    for (int w=0; w<W; w++)
      {
      beta_vec[w] = beta;  
      }
       
    for (int k=0;k<K;k++)
      { 
      phi.row(k) = rDirichlet(beta_vec, W);
      }
    return phi;
    }    
     

void LDA::NichollsMH(int iter, int burnin, int thin)
  {

    arma::mat phi_current = DrawFromProposalInit();
    
    for (int t=1;t<iter;t++)
    {

    // Metropolis-Hastings Algorithm:
    // 1. draw from proposal density:
    arma::mat hyperParams = beta + 0.1 * (phi_current - beta);
    arma::mat phi_new = DrawFromProposal(hyperParams);

    // 2. Calculate acceptance probability
    double pi_new = PhiDensity(phi_new);
    double pi_old = PhiDensity(phi_current);
    double q_new = ProposalDensity(phi_new);
    double q_old = ProposalDensity(phi_current);

    double acceptanceMH = exp(pi_new + q_old - pi_old - q_new);
    double alphaMH = min((double)1,acceptanceMH);
    Rcout << "Acceptance Prob:" << alphaMH;

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
  
double LDA::LogPhiProd(arma::mat phi)
  {
  arma::mat logPhi = log(phi);
  double sumLik_vec[K];
  double logPhiProd = 0;
    
    for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       sumLik_vec[k] = dot(logPhi_k,nd);
       }
     double b = ArrayMax(sumLik_vec,K);
     
     for (int k=0; k<K; k++)
       {
       sumLik += exp(sumLik_vec[k]-b);
       }
     
     logPhiProd += b + log(sumLik);
     }  
     
  return logPhiProd;  
  }
  
vector<double> LDA::LogPhiProd_vec(arma::mat phi)
  {
  vector<double> ret_vec;
  arma::mat logPhi = log(phi);
  double sumLik_vec[K];
    
    for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       sumLik_vec[k] = dot(logPhi_k,nd);
       PhiProd_vec.push_back(sumLik_vec[k]);
       }
     double b = ArrayMax(sumLik_vec,K);
     
     for (int k=0; k<K; k++)
       {
       sumLik += exp(sumLik_vec[k]-b);
       }
     
     double ret_vec_d = b + log(sumLik);
     ret_vec.push_back(ret_vec_d);
     }  
     
  return ret_vec;  
  }
    
  
  
NumericMatrix LDA::PhiGradient(NumericMatrix phi)
  {
    double sumLik_vec[K];
    arma::mat phi2 = as<arma::mat>(phi);
    arma::mat logPhi = log(phi2);
    vector<double> denom_vec = LogPhiProd_vec(phi2);
    arma::mat gradient(K,W);
    
    for (int z=0;z<K;z++)
      {
      for(int w=0;w<W;w++)
        {
        double dotProd = PhiProd_vec[z];
        Rcout << "dotProd:" << dotProd;
        double dSum = 0;  
        
        for (int d = 0; d<D;d++)
          {  
          double nwd = n_wd(w,d);
          if (nwd==0) dSum += 0;
          else 
            {
            // Rcout << "nwd:" << nwd;
            arma::colvec nd = n_wd.col(d);
            arma::rowvec logPhi_k = logPhi.row(z);
            
            // Rcout << "Dot Product: " << dotProd;
            
            double Numerator = log(nwd) + (nwd - 1)*logPhi(z,w) + dotProd - nd[w]*logPhi_k[w]; 
            Rcout << Numerator;
            double Denominator = denom_vec[d];
            Rcout << Denominator;
            dSum += exp(Numerator - Denominator);
            }
          }
        // Rcout << "dSum:" << dSum; 
        gradient(z,w) = dSum + (beta - 1) / phi2(z,w);
        //Rcout << gradient(z,w);
        }
      }
       
    return wrap(gradient);   
  }  
  
  

double LDA::ProposalDensity(arma::mat phi)
  {
    double logBetaFun = 0;
    double betaSum = 0;
    for (int k=0; k<K;k++)
      {
      for (int w=0;w<W;w++)
        {
        double phi_scalar = phi(k,w);
        logBetaFun += lgamma(phi_scalar);
        betaSum  += phi_scalar;
        }
          
      }
    // double logBetaFun = K*(W*lgamma(beta)-lgamma(W*beta));
    logBetaFun -= lgamma(betaSum);
    
    arma::mat logPhi = log(phi);
    arma::mat temp = logPhi * (beta-1);
    double logPhiSum = accu(temp);

    double logDensity = logPhiSum - logBetaFun;
    return logDensity;
  }

double LDA::PhiDensity(arma::mat phi)
  {
  arma::mat logPhi = log(phi);
  double logLikelihood_vec[D];
  double sumLik_vec[K];
  double logLikelihood = 0;

   for (int d=0; d<D; d++)
     {
     double sumLik = 0;
     arma::colvec nd = n_wd.col(d);

     for (int k=0; k<K; k++)
       {
       arma::rowvec logPhi_k = logPhi.row(k);
       sumLik_vec[k] = dot(logPhi_k,nd);
       sumLik += sumLik_vec[k];
       }
     double b = ArrayMax(sumLik_vec,K);
     logLikelihood_vec[d] = exp(sumLik + b);
     // logLikelihood += b + log(logLikelihood_vec[d]);
     logLikelihood += b;
     }
     
      logLikelihood += D * log(alpha);
      logLikelihood -= D * log(K*alpha);
      // Rcout << "logLikelihood: " << logLikelihood;

      double logBetaFun = K*(lgamma(W*beta)-W*lgamma(beta));
      // Rcout << "logBetaFun: " << logBetaFun;

      double logPhiSum = 0;

      arma::mat temp = logPhi * (beta-1);
      logPhiSum = accu(temp);

      // Rcout << "LogPhiSum: " << logPhiSum;

    double logProb = logLikelihood + logBetaFun + logPhiSum;
    return logProb;
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


double LDA::ArrayMax(double array[], int numElements)
{
     double max = array[0];       // start with max = first element

     for(int i = 1; i<numElements; i++)
     {
          if(array[i] > max)
                max = array[i];
     }
     return max;                // return highest value in array
}

double LDA::ArrayMin(double array[], int numElements)
{
     double min = array[0];       // start with min = first element

     for(int i = 1; i<numElements; i++)
     {
          if(array[i] < min)
                min = array[i];
     }
     return min;                // return smallest value in array
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
.method("rgamma_cpp",&LDA::rgamma_cpp)
.method("rbeta_cpp",&LDA::rbeta_cpp)
.method("DrawFromProposalInit",&LDA::DrawFromProposalInit)
.method("rDirichlet2",&LDA::rDirichlet2)
.method("LogPhiProd_vec",&LDA::LogPhiProd_vec)
;
}
