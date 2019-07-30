#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

#include <cmath>

#include "../Lib/LKH3Lib/TspSolver.h"


using namespace std;


namespace szx {

#pragma region Solver::Cli
int Solver::Cli::run(int argc, char * argv[]) {
    Log(LogSwitch::Szx::Cli) << "parse command line arguments." << endl;
    Set<String> switchSet;
    Map<String, char*> optionMap({ // use string as key to compare string contents instead of pointers.
        { InstancePathOption(), nullptr },
        { SolutionPathOption(), nullptr },
        { RandSeedOption(), nullptr },
        { TimeoutOption(), nullptr },
        { MaxIterOption(), nullptr },
        { JobNumOption(), nullptr },
        { RunIdOption(), nullptr },
        { EnvironmentPathOption(), nullptr },
        { ConfigPathOption(), nullptr },
        { LogPathOption(), nullptr }
    });

    for (int i = 1; i < argc; ++i) { // skip executable name.
        auto mapIter = optionMap.find(argv[i]);
        if (mapIter != optionMap.end()) { // option argument.
            mapIter->second = argv[++i];
        } else { // switch argument.
            switchSet.insert(argv[i]);
        }
    }

    Log(LogSwitch::Szx::Cli) << "execute commands." << endl;
    if (switchSet.find(HelpSwitch()) != switchSet.end()) {
        cout << HelpInfo() << endl;
    }

    if (switchSet.find(AuthorNameSwitch()) != switchSet.end()) {
        cout << AuthorName() << endl;
    }

    Solver::Environment env;
    env.load(optionMap);
    if (env.instPath.empty() || env.slnPath.empty()) { return -1; }

    Solver::Configuration cfg;
    cfg.load(env.cfgPath);

    Log(LogSwitch::Szx::Input) << "load instance " << env.instPath << " (seed=" << env.randSeed << ")." << endl;
    Problem::Input input;
    if (!input.load(env.instPath)) { return -1; }

    Solver solver(input, env, cfg);
    solver.solve();

    pb::Submission submission;
    submission.set_thread(to_string(env.jobNum));
    submission.set_instance(env.friendlyInstName());
    submission.set_duration(to_string(solver.timer.elapsedSeconds()) + "s");

    solver.output.save(env.slnPath, submission);
    #if SZX_DEBUG
    solver.output.save(env.solutionPathWithTime(), submission);
    solver.record();
    #endif // SZX_DEBUG

    return 0;
}
#pragma endregion Solver::Cli

#pragma region Solver::Environment
void Solver::Environment::load(const Map<String, char*> &optionMap) {
    char *str;

    str = optionMap.at(Cli::EnvironmentPathOption());
    if (str != nullptr) { loadWithoutCalibrate(str); }

    str = optionMap.at(Cli::InstancePathOption());
    if (str != nullptr) { instPath = str; }

    str = optionMap.at(Cli::SolutionPathOption());
    if (str != nullptr) { slnPath = str; }

    str = optionMap.at(Cli::RandSeedOption());
    if (str != nullptr) { randSeed = atoi(str); }

    str = optionMap.at(Cli::TimeoutOption());
    if (str != nullptr) { msTimeout = static_cast<Duration>(atof(str) * Timer::MillisecondsPerSecond); }

    str = optionMap.at(Cli::MaxIterOption());
    if (str != nullptr) { maxIter = atoi(str); }

    str = optionMap.at(Cli::JobNumOption());
    if (str != nullptr) { jobNum = atoi(str); }

    str = optionMap.at(Cli::RunIdOption());
    if (str != nullptr) { rid = str; }

    str = optionMap.at(Cli::ConfigPathOption());
    if (str != nullptr) { cfgPath = str; }

    str = optionMap.at(Cli::LogPathOption());
    if (str != nullptr) { logPath = str; }

    calibrate();
}

void Solver::Environment::load(const String &filePath) {
    loadWithoutCalibrate(filePath);
    calibrate();
}

void Solver::Environment::loadWithoutCalibrate(const String &filePath) {
    // EXTEND[szx][8]: load environment from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Environment::save(const String &filePath) const {
    // EXTEND[szx][8]: save environment to file.
}
void Solver::Environment::calibrate() {
    // adjust thread number.
    int threadNum = thread::hardware_concurrency();
    if ((jobNum <= 0) || (jobNum > threadNum)) { jobNum = threadNum; }

    // adjust timeout.
    msTimeout -= Environment::SaveSolutionTimeInMillisecond;
}
#pragma endregion Solver::Environment

#pragma region Solver::Configuration
void Solver::Configuration::load(const String &filePath) {
    // EXTEND[szx][5]: load configuration from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Configuration::save(const String &filePath) const {
    // EXTEND[szx][5]: save configuration to file.
}
#pragma endregion Solver::Configuration

#pragma region Solver
bool Solver::solve() {
    init();

    int workerNum = (max)(1, env.jobNum / cfg.threadNumPerWorker);
    cfg.threadNumPerWorker = env.jobNum / workerNum;
    List<Solution> solutions(workerNum, Solution(this));
    List<bool> success(workerNum);

    Log(LogSwitch::Szx::Framework) << "launch " << workerNum << " workers." << endl;
    List<thread> threadList;
    threadList.reserve(workerNum);
    for (int i = 0; i < workerNum; ++i) {
        // TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
        // OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
        threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], i); });
    }
    for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

    Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
    int bestIndex = -1;
    Price bestValue = (numeric_limits<Price>::max)();
    for (int i = 0; i < workerNum; ++i) {
        if (!success[i]) { continue; }
        Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].cost << endl;
        if (solutions[i].cost >= bestValue) { continue; }
        bestIndex = i;
        bestValue = solutions[i].cost;
    }

    env.rid = to_string(bestIndex);
    if (bestIndex < 0) { return false; }
    output = solutions[bestIndex];
    return true;
}

void Solver::record() const {
    #if SZX_DEBUG
    int generation = 0;

    ostringstream log;

    System::MemoryUsage mu = System::peakMemoryUsage();

    Price obj = output.cost;
    Price checkerObj = -1;
    bool feasible = check(checkerObj);

    // record basic information.
    log << env.friendlyLocalTime() << ","
        << env.rid << ","
        << env.instPath << ","
        << feasible << "," << (obj - checkerObj) << ","
        << obj << ","
        << timer.elapsedSeconds() << ","
        << mu.physicalMemory << "," << mu.virtualMemory << ","
        << env.randSeed << ","
        << cfg.toBriefStr() << ","
        << generation << "," << iteration << ",";

    // record solution vector.
    for (auto n = output.path().begin(); n != output.path().end(); ++n) {
        log << n->node() << " ";
    }
    log << endl;

    // append all text atomically.
    static mutex logFileMutex;
    lock_guard<mutex> logFileGuard(logFileMutex);

    ofstream logFile(env.logPath, ios::app);
    logFile.seekp(0, ios::end);
    if (logFile.tellp() <= 0) {
        logFile << "Time,ID,Instance,Feasible,ObjMatch,Color,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
    }
    logFile << log.str();
    logFile.close();
    #endif // SZX_DEBUG
}

bool Solver::check(Price &checkerObj) const {
    #if SZX_DEBUG
    enum CheckerFlag {
        IoError = 0x0,
        FormatError = 0x1,
        DemandUnsatisfiedError = 0x2,
        NotEnoughSupplyError = 0x4,
        NonExistingEdgeError = 0x8,
        NotSimplePathError = 0x10
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::DemandUnsatisfiedError) { Log(LogSwitch::Checker) << "DemandUnsatisfiedError." << endl; }
    if (checkerObj & CheckerFlag::NotEnoughSupplyError) { Log(LogSwitch::Checker) << "NotEnoughSupplyError." << endl; }
    if (checkerObj & CheckerFlag::NonExistingEdgeError) { Log(LogSwitch::Checker) << "NonExistingEdgeError." << endl; }
    if (checkerObj & CheckerFlag::NotSimplePathError) { Log(LogSwitch::Checker) << "NotSimplePathError." << endl; }
    return false;
    #else
    checkerObj = 0;
    return true;
    #endif // SZX_DEBUG
}

void Solver::init() {
    ID nodeNum = input.graph().nodes().size();

    aux.adjList.resize(nodeNum);
    aux.adjMat.init(nodeNum, nodeNum);
    aux.adjMat.reset(Arr2D<Price>::ResetOption::SafeMaxInt); // TODO[szx][9]: make sure the `Price` is integer type, or use the following line to init.
    //fill(aux.adjMat.begin(), aux.adjMat.end(), Problem::MaxCost);
    aux.maxEdgeCost = 0;
    for (auto e = input.graph().edges().begin(); e != input.graph().edges().end(); ++e) {
        // assume there is no duplicated edge.
        aux.adjList.at(e->src()).push_back(e->dst());
        aux.adjMat.at(e->src(), e->dst()) = e->cost();
        if (aux.maxEdgeCost < e->cost()) { aux.maxEdgeCost = e->cost(); }
    }
}

bool Solver::optimize(Solution &sln, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;

    // reset solution state.
    bool status = true;

    //status = optimizeByRandomAssignment(sln);
    status = optimizeShortestSimplePathWithMustPassNodesByReduction(sln);

    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}

bool Solver::optimizeByRandomAssignment(Solution &sln) {
    ID nodeNum = input.graph().nodes().size();
    auto &path(*sln.mutable_path());

    Price routingCost = 0;
    Price purchaseCost = 0;
    ID prevNode = input.src();
    while (!timer.isTimeOut()) {
        pb::TravelingPurchaser::Purchase &purchase(*path.Add());
        purchase.set_node(rand.pick(nodeNum));
        routingCost += aux.adjMat.at(prevNode, purchase.node());
        if (purchase.node() == input.dst()) { break; }
        auto &quantities(*purchase.mutable_quantities());
        auto &supplies(input.graph().nodes(purchase.node()).supplies());
        for (auto p = supplies.begin(); p != supplies.end(); ++p) {
            quantities[p->first] = p->second.quantity();
            purchaseCost += (p->second.quantity() * p->second.price());
        }
        prevNode = purchase.node();
    }

    sln.cost = routingCost + purchaseCost; // record obj.
    return true;
}

bool Solver::optimizeShortestSimplePathWithMustPassNodesByReduction(Solution &sln) {
    Log(LogSwitch::Szx::Reduction) << "reduction starts." << endl;
    ID nodeNum = input.graph().nodes().size();

    List<ID> productProviders(input.demands().size(), Problem::InvalidId);
    List<ID> providedProducts(input.graph().nodes().size(), Problem::InvalidId);

    ConsecutiveIdSet mustPassNodes(nodeNum);
    for (ID n = 0; n < nodeNum; ++n) {
        auto &node(input.graph().nodes(n));
        if (node.supplies().empty()) { continue; }
        auto &supply(*node.supplies().begin());

        if ((node.supplies().size() > 1) // provide single product.
            || (supply.second.quantity() < input.demands(supply.first)) // fulfill the demand at single visit.
            || (productProviders[supply.first] > Problem::InvalidId)) { // no one else provides the same product.
            Log(LogSwitch::Szx::Reduction) << "[error] this problem can not be regarded as shortest simple path problem with must-pass nodes." << endl;
            return false;
        }
        productProviders[supply.first] = n;
        providedProducts[n] = supply.first;

        mustPassNodes.insert(n);
    }

    ID virtualNodeNum = (2 * nodeNum) - mustPassNodes.size() - 1;

    Arr2D<Price> adjMat(virtualNodeNum, virtualNodeNum);
    adjMat.reset(Arr2D<Price>::ResetOption::SafeMaxInt); // TODO[szx][9]: make sure the `Price` is integer type, or use the following line to init.
    //fill(adjMat.begin(), adjMat.end(), Problem::MaxCost);

    // copy original adjacency information.
    Price costAmp = virtualNodeNum;
    for (ID n = 0; n < nodeNum; ++n) {
        for (auto m = aux.adjList[n].begin(); m != aux.adjList[n].end(); ++m) {
            adjMat.at(n, *m) = aux.adjMat.at(n, *m) * costAmp;
        }
    }
    //for (ID n = 0; n < nodeNum; ++n) {
    //    for (ID m = 0; m < nodeNum; ++m) {
    //        adjMat.at(n, m) = aux.adjMat.at(n, m);
    //    }
    //}

    // add virtual nodes and edges.
    ID virtualNodeId = nodeNum;
    ID nextVirtualNodeId = nodeNum + 1;
    adjMat.at(input.dst(), virtualNodeId) = 1;
    for (ID n = 0; n < nodeNum; ++n) {
        if (mustPassNodes.isItemExist(n) || (n == input.src()) || (n == input.dst())) { continue; }
        adjMat.at(virtualNodeId, nextVirtualNodeId) = 1;
        adjMat.at(virtualNodeId, n) = 1;
        adjMat.at(n, nextVirtualNodeId) = 1;
        ++virtualNodeId;
        ++nextVirtualNodeId;
    }
    adjMat.at(virtualNodeId, input.src()) = 1;

    Log(LogSwitch::Szx::Reduction) << "start solving reduced problem." << endl;
    lkh::Tour tour;
    if (lkh::solveTsp(tour, adjMat)) {
        Log(LogSwitch::Szx::Reduction) << "retrieving solution." << endl;
        sln.cost = tour.distance / costAmp; // record obj.

        auto &path(*sln.mutable_path());
        auto n = tour.nodes.begin();
        bool isDstBeforeSrc = false;
        for (; *n != input.src(); ++n) { isDstBeforeSrc |= (*n == input.dst()); }
        if (isDstBeforeSrc) {
            rotate(tour.nodes.begin(), n, tour.nodes.end());
            n = tour.nodes.begin();
        }
        for (ID prevNode = input.src(); prevNode != input.dst(); prevNode = *n, ++n) {
            pb::TravelingPurchaser::Purchase &purchase(*path.Add());
            purchase.set_node(*n);
            if (providedProducts[*n] <= Problem::InvalidId) { continue; }
            auto &quantities(*purchase.mutable_quantities());
            quantities[providedProducts[*n]] = input.demands(providedProducts[*n]);
        }
        return true;
    }

    return false;
}
#pragma endregion Solver

}
