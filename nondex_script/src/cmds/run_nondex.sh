#!/bin/bash

projectRootDir=$1  
module_path_relative_to_root=$2 
test_to_run=$3
jdk_version=$4
nondex_runs_count=$5

mainDir=${projectRootDir} 
curDir=$(pwd) 

actual_module_dir=${projectRootDir} 
if [[ -n "${module_path_relative_to_root}" && "${module_path_relative_to_root}" != "." && "${module_path_relative_to_root}" != '""' && "${module_path_relative_to_root}" != "root" ]]; then
    actual_module_dir=${projectRootDir}/${module_path_relative_to_root}
fi

run_nondex_in_module(){
    echo "[run_nondex.sh] Executing: mvn edu.illinois:nondex-maven-plugin:2.1.7:nondex -Dtest=${test_to_run} -DnondexRuns=${nondex_runs_count} ..."
    
    mvn edu.illinois:nondex-maven-plugin:2.1.7:nondex \
        -Dtest=${test_to_run} \
        -Dbasepom.check.skip-prettier -Dgpg.skip -Dfindbugs.skip=true -Drat.skip \
        -Dcheckstyle.skip -Denforcer.skip=true -Dspotbugs.skip \
        -Dmaven.test.failure.ignore=true -Djacoco.skip -Danimal.sniffer.skip \
        -Dmaven.antrun.skip -Dfmt.skip -Dskip.npm -Dlicense.skipCheckLicense \
        -Dlicense.skipAddThirdParty=true -Dfindbugs.skip -Dlicense.skip \
        -Dskip.npm -Dskip.yarn -Dskip.bower -Dskip.grunt -Dskip.gulp \
        -Dskip.jspm -Dskip.karma -Dskip.webpack -DskipDockerBuild \
        -DskipDockerTag -DskipDockerPush -DskipDocker -Denforcer.skip \
        -DnondexRuns=${nondex_runs_count} \
        -Dstyle.color=never -Ddependency-check.skip -Dspotless.check.skip
}

echo "[run_nondex.sh] RUNNING NonDex on ID test ${test_to_run} in module path ${module_path_relative_to_root} (Plugin runs: ${nondex_runs_count}) STARTING at $(date)"
echo "[run_nondex.sh] REPO VERSION (from ${mainDir}): $(cd "${mainDir}" && git rev-parse HEAD)"

if [[ ! -d "${actual_module_dir}" ]]; then
    echo "[run_nondex.sh] ERROR: Module directory '${actual_module_dir}' does not exist. Aborting."
    exit 1
fi

cd "${actual_module_dir}" 

echo "[run_nondex.sh] CURRENT DIR for mvn execution: $(pwd)"
echo "[run_nondex.sh] Expected Java version ${jdk_version}"

if  [[ ${jdk_version} == "8" ]]; then
    echo "[run_nondex.sh] Java version 8"
    export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64 
    export PATH=$JAVA_HOME/bin:$PATH
elif  [[ ${jdk_version} == "11" ]]; then
    echo "[run_nondex.sh] Java version 11"
    export JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64 
    export PATH=$JAVA_HOME/bin:$PATH
else
    echo "[run_nondex.sh] Warning: Unknown JDK version '${jdk_version}' specified. Using system default."
fi

run_nondex_in_module
exit_code=$? 

cd "${curDir}"
echo "[run_nondex.sh] RUNNING NonDex on ID test ${test_to_run} ENDING at $(date)"
exit ${exit_code}