#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

#include <htm/os/Timer.hpp>
#include <htm/types/Types.hpp>

#include "htm/algorithms/AnomalyLikelihood.hpp"
#include "htm/algorithms/SpatialPooler.hpp"
#include "htm/algorithms/TemporalMemory.hpp"
#include "htm/encoders/RandomDistributedScalarEncoder.hpp"

#include "htm/types/Sdr.hpp"
#include "htm/utils/MovingAverage.hpp"
#include "htm/utils/Random.hpp"
#include "htm/utils/SdrMetrics.hpp"

#include <opencv2/opencv.hpp>

#include <nlohmann/json.hpp>

using htm::Real64;
using htm::SpatialPooler;
using htm::TemporalMemory;
using htm::Timer;
using htm::UInt;

using json = nlohmann::json;

using namespace std;
using namespace htm;
using namespace cv;

#include <string> // stoi

void renderSDR(SDR& sdr, string kind, int epoch) {
    auto &dense = sdr.getDense();

    // Create an image with 128 rows and 128 columns
    Mat img(128, 128, CV_8UC1);

    // Copy the vector values to the image
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (dense[i * img.cols + j] == 0) {
                img.at<uchar>(i, j) = 0;
            } else {
                img.at<uchar>(i, j) = 255;
            }
        }
    }

    auto epoch_num = to_string(epoch);
    size_t n_zero = 4;
    auto epoch_str = std::string(n_zero - std::min(n_zero, epoch_num.length()), '0') + epoch_num;

    // Save the image to file
    string filename = "states/" + kind + epoch_str + ".png";

    Mat resized_up;

    cv::resize(img, resized_up, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR_EXACT);

    imwrite(filename, resized_up);
}

map<string, vector<int>> read_terms(const string& file_path) {
    map<string, vector<int>> result;
    ifstream file(file_path);

    if (!file) {
        throw runtime_error("Cannot open file: " + file_path);
    }

    string line;

    int i = 0;

    while (getline(file, line)) {
        json j_line = json::parse(line);
        string term = j_line["term"];
        vector<int> positions = j_line["fingerprint"]["positions"];
        result[term] = positions;

        if (i % 100000 == 0) {
            cout << "terms " << i << endl;
        }

        i++;
    }

    return result;
}

// this runs as executable:  hello [epochs]
int main(int argc, char *argv[]) {
    auto terms = read_terms("terms.txt");

    std::string filename = "snip.jsonl";

    vector<vector<string>> data;

    std::ifstream file(filename);
    std::string line;

    int i = 0;

    if (std::filesystem::exists(filename)) {
        while (std::getline(file, line)) {
            try {
                vector<string> sentence = json::parse(line);

                data.push_back(sentence);
            } catch (json::parse_error &e) {
                // output exception information
                std::cout << "message: " << e.what() << '\n'
                          << "exception id: " << e.id << '\n'
                          << "byte position of error: " << e.byte << std::endl;
                return EXIT_FAILURE;
            } catch (json::type_error &e) {
                // output exception information
                std::cout << "message: " << e.what() << '\n'
                          << "exception id: " << e.id << '\n'
                          << std::endl;
                return EXIT_FAILURE;
            }

            i++;

            if (i % 10000 == 0) {
                std::cout << i << std::endl;
            }
        }
    } else {
        std::cout << "File '" << filename << "' does not exist." << std::endl;
        return EXIT_FAILURE;
    }

    const UInt SP_COLLS = 16384;
    //const UInt TM_COLLS = 2048;
    const UInt TM_COLLS = 16384;

    // Create the HTM network
    SpatialPooler sp(vector<UInt>{SP_COLLS}, vector<UInt>{TM_COLLS});
    TemporalMemory tm(vector<UInt>{TM_COLLS}, 64);

    sp.printParameters();

    SDR test_word(vector<UInt>{SP_COLLS});
    test_word.setDense(terms["while"]);

    sp.setGlobalInhibition(true);

    //SDR outSP(vector<UInt>{TM_COLLS});
    //SDR outTM(sp.getColumnDimensions());

    i = 0;

    // Train the network on each sentence
    for (auto sentence : data) {
        SDR outSP(vector<UInt>{TM_COLLS});

        // Create an SDR for the sentence
        std::vector<int> sentence_sdr(SP_COLLS, 0);

        std::vector<Real64> anomaly_likelihoods;

        // Train the network on each word in the sentence
        for (auto word : sentence) {
            // Get the word and its positions in the SDR
            std::vector<int> positions = terms[word];

            // Create an SDR for the word
            std::vector<UInt> word_vec(SP_COLLS, 0);
            for (auto position : positions) {
                word_vec[position] = 1;
            }

            SDR word_sdr(vector<UInt>{SP_COLLS});

            word_sdr.setDense(word_vec);

            // Train the spatial pooler on the word SDR
            sp.compute(word_sdr, true, outSP);

            // Train the temporal memory on the spatial pooler output
            tm.compute(outSP, true);

            // Compute the anomaly likelihood
            Real64 anomaly_likelihood = tm.anomaly;

            anomaly_likelihoods.push_back(anomaly_likelihood);

            //tm.activateDendrites(
            //    true
            //);
            //
            //outTM = tm.cellsToColumns(tm.getPredictiveCells());
        }

        // Print the sentence
        std::cout << i << ". Sentence: ";

        for (auto const& word : sentence) {
            std::cout << word << " ";
        }

        std::cout << std::endl;

        // accumulate the anomaly likelihoods
        Real64 anomaly_likelihood_sum = 0.0;

        for (auto anomaly_likelihood : anomaly_likelihoods) {
            anomaly_likelihood_sum += anomaly_likelihood;
        }

        // compute the average anomaly likelihood
        Real64 anomaly_likelihood_avg = anomaly_likelihood_sum / anomaly_likelihoods.size();

        // print the anomaly likelihoods
        std::cout << "Anomaly likelihoods: ";

        for (auto anomaly_likelihood : anomaly_likelihoods) {
            std::cout << anomaly_likelihood << " ";
        }

        std::cout << std::endl;

        // print the average anomaly likelihood
        std::cout << "Anomaly likelihood avg: " << anomaly_likelihood_avg << std::endl;

        // Print the spatial pooler output
        renderSDR(outSP, "sp", i);

        // Print test
        SDR outSP_test(vector<UInt>{TM_COLLS});
        sp.compute(test_word, false, outSP_test);
        renderSDR(outSP_test, "sp_test", i);

//        std::cout << "SP: ";
//        for (auto bit : outSP.getSparse()) {
//            std::cout << bit << " ";
//        }
//        std::cout << std::endl;
//
//        // Print the temporal memory output
//        renderSDR(outTM, "tm", i);
//
//        std::cout << "TM: ";
//        for (auto bit : outTM.getSparse()) {
//            std::cout << bit << " ";
//        }
//        std::cout << std::endl;
//
//        // Print the predictive cells
//        tm.activateDendrites(false);
//
//        SDR pred_cells = tm.getPredictiveCells();
//        renderSDR(pred_cells, "pred", i);

        tm.reset();

        i += 1;

        if (i % 1000 == 0) {
            sp.saveToFile("sp.json", SerializableFormat::JSON);
                tm.saveToFile("tm.json", SerializableFormat::JSON);
        }
    }

    return 0;
}
