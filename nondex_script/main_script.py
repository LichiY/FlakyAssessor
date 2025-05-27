import csv
import os
import subprocess
import re
import logging
import pandas as pd
from pathlib import Path
import shutil
import sys
import json
FLAKYDOCTOR_SRC_PATH = "./src"
if FLAKYDOCTOR_SRC_PATH not in sys.path:
    abs_fd_src_path = os.path.abspath(FLAKYDOCTOR_SRC_PATH)
    if os.path.isdir(abs_fd_src_path):
        sys.path.insert(0, abs_fd_src_path)
    else:
        print(f"ERROR: FLAKYDOCTOR_SRC_PATH '{abs_fd_src_path}' does not exist or is not a directory.")
        sys.exit(1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler("run.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
try:
    import utils as fd_utils
    from update_pom import add_dependency as fd_add_dependency
except ImportError as e:
    logger.error(f"Error importing FlakyDoctor modules: {e}", exc_info=True)
    logger.error(f"Please ensure FLAKYDOCTOR_SRC_PATH ('{FLAKYDOCTOR_SRC_PATH}') is correct.")
    sys.exit(1)
FLAKY_REPOS_DIR = "./flaky_repos"
PATCH_BASE_DIR = "./patch"
OUTPUT_DIR = "."
MERGED_DATASET_CSV = "complete_merged_dataset_2.csv"
MERGED_IMPORT_POM_CSV = os.path.join(OUTPUT_DIR, "merged_import_pom.csv")
NONDEX_VALIDATION_RESULTS_CSV = os.path.join(OUTPUT_DIR, "nondex_validation_results.csv")
PATCHED_JAVA_FILES_DIR = os.path.join(OUTPUT_DIR, "patched_java_file")
DEBUG_JAVA_FILES_DIR = os.path.join(OUTPUT_DIR, "debug_java_files")
FD_RUN_NONDEX_CMD = os.path.join(os.path.abspath(FLAKYDOCTOR_SRC_PATH), "cmds/run_nondex.sh")
JAVA_STANDARD_LIBS_PATH = os.path.join(os.path.abspath(FLAKYDOCTOR_SRC_PATH), "utils/java_standard_libs.json")
NONDEX_INTERNAL_RUNS_COUNT = 5
java_standard_libs_data = {}
if os.path.exists(JAVA_STANDARD_LIBS_PATH):
    try:
        with open(JAVA_STANDARD_LIBS_PATH, 'r', encoding='utf-8') as f_libs: java_standard_libs_data = json.load(f_libs)
    except json.JSONDecodeError as e: logger.error(f"ERROR: Could not decode {JAVA_STANDARD_LIBS_PATH}: {e}"); java_standard_libs_data = {}
else: logger.warning(f"WARNING: {JAVA_STANDARD_LIBS_PATH} not found. Import stitching will be limited.")
def find_flakydoctor_results_csv(patch_sub_dir):
    if not os.path.isdir(patch_sub_dir): logger.warning(f"Dir for FD results CSV not found: {patch_sub_dir}"); return None
    for f_name in os.listdir(patch_sub_dir):
        if f_name.endswith(".csv") and ("MagicCoder_results" in f_name or "GPT-4_results" in f_name): return os.path.join(patch_sub_dir, f_name)
    logger.warning(f"No FD results CSV found in {patch_sub_dir}"); return None
def run_git_command(project_dir, command_list, check=True, timeout=300):
    cwd = os.getcwd()
    try:
        if not os.path.isdir(project_dir): logger.error(f"Project dir '{project_dir}' invalid for git cmd: git {' '.join(command_list)}."); return False, "ProjectDirNotFound", f"Project dir '{project_dir}' invalid."
        os.chdir(project_dir)
        process = subprocess.run(["git"] + command_list, capture_output=True, text=True, check=check, timeout=timeout, errors='replace')
        os.chdir(cwd); return True, process.stdout, process.stderr
    except subprocess.CalledProcessError as e: logger.error(f"Git cmd git {' '.join(command_list)} failed in {project_dir}: {e.stderr.strip() if e.stderr else e.stdout.strip()}"); os.chdir(cwd); return False, e.stdout, e.stderr
    except subprocess.TimeoutExpired: logger.error(f"Git cmd git {' '.join(command_list)} timed out in {project_dir}"); os.chdir(cwd); return False, "Timeout", "Timeout"
    except FileNotFoundError: logger.error(f"Git cmd not found or dir '{project_dir}' invalid for git {' '.join(command_list)}.");
    except Exception as e: logger.error(f"Exception during git cmd git {' '.join(command_list)} in {project_dir}: {e}", exc_info=True);
    finally:
        if os.getcwd() != cwd:
            try: os.chdir(cwd)
            except Exception as e_chdir: logger.error(f"Failed to chdir back to {cwd}: {e_chdir}")
    return False, "UnknownGitError", "Unknown error in run_git_command"
def convert_to_ssh_url(http_url):
    if isinstance(http_url, str) and http_url.startswith("https://github.com/"): return http_url.replace("https://github.com/", "git@github.com:")
    return http_url
def git_clone_project(repo_url, project_dir_parent, project_name_in_csv):
    project_full_path = os.path.join(project_dir_parent, project_name_in_csv)
    if os.path.isdir(project_full_path):
        logger.info(f"Project {project_name_in_csv} exists. Removing for fresh clone: {project_full_path}")
        try: subprocess.run(f"rm -rf \"{project_full_path}\"", shell=True, check=True, capture_output=True); logger.info(f"Removed: {project_full_path}")
        except Exception as e: logger.error(f"Failed to remove {project_full_path}: {e}. Pls remove manually."); return False
    logger.info(f"Cloning {repo_url} into {project_full_path}...")
    Path(project_dir_parent).mkdir(parents=True, exist_ok=True)
    ssh_repo_url = convert_to_ssh_url(repo_url)
    logger.info(f"Attempting clone via SSH: {ssh_repo_url}")
    clone_command = ["clone", "--no-tags", ssh_repo_url, project_full_path]
    try:
        logger.info(f"Running: git {' '.join(clone_command)}")
        subprocess.run(["git"] + clone_command, capture_output=True, text=True, check=True, timeout=1800, errors='replace')
        logger.info(f"Cloned {ssh_repo_url} to {project_full_path}"); return True
    except Exception as e: logger.error(f"Git clone (SSH) failed for {ssh_repo_url}: {e}"); return False
def git_checkout_commit(project_dir, sha, repo_url, project_name_in_csv):
    if not os.path.isdir(os.path.join(project_dir, ".git")):
        logger.warning(f"{project_dir} not a git repo. Cloning...")
        if not git_clone_project(repo_url, FLAKY_REPOS_DIR, project_name_in_csv): logger.error(f"Clone failed for {repo_url}."); return False
        project_dir = os.path.join(FLAKY_REPOS_DIR, project_name_in_csv)
    logger.info(f"Fetching to ensure SHA {sha} is available in {project_dir}")
    run_git_command(project_dir, ["fetch", "origin", "--prune", "--no-tags", "--unshallow"], check=False, timeout=1800)
    fetch_ok, _, fetch_err = run_git_command(project_dir, ["fetch", "origin", sha], check=False, timeout=600)
    run_git_command(project_dir, ["stash", "push", "-u", "-m", "apply_script_stash"], check=False)
    success, _, stderr = run_git_command(project_dir, ["checkout", sha])
    if not success:
        logger.warning(f"Initial git checkout to {sha} failed in {project_dir}. Error: {stderr.strip()}")
        if any(msg in stderr for msg in ["您尚未建立初始提交", "does not exist", "reference is not a tree", "did not match any file(s) known to git", "isn't a commit"]):
            logger.error(f"SHA {sha} seems invalid or not found in the repository {repo_url} even after extensive fetching.")
            return False
        else: return False
    logger.info(f"Successfully checked out SHA {sha} in {project_dir}"); return True
def git_clean_repo(project_dir):
    logger.info(f"Cleaning repo {project_dir}")
    r1,_,_ = run_git_command(project_dir, ["reset", "--hard", "HEAD"], check=False)
    r2,_,_ = run_git_command(project_dir, ["clean", "-fdx"], check=False)
    return r1 and r2
def get_fqn_parts_from_input_csv(full_test_name_str, source_file_type):
    if not isinstance(full_test_name_str, str): return None, None
    parts = full_test_name_str.split('.')
    if not parts: return None, None
    if source_file_type == "flaky_doctor_131_gpt_fixed_checked_labeled_time.csv":
        if len(parts) >= 3 and parts[-1].lower() == parts[-2].lower():
            method_name = parts[-1]
            simple_class_name = parts[-3]
            package_parts = parts[:-3]
            class_qualifier = ".".join(package_parts + [simple_class_name]) if package_parts else simple_class_name
            return class_qualifier, method_name
        elif len(parts) >= 2:
            method_name = parts[-1]; class_qualifier = ".".join(parts[:-1])
            return class_qualifier, method_name
        else: return None, parts[0] if parts else None
    else:
        if len(parts) >= 2:
            method_name = parts[-1]; class_qualifier = ".".join(parts[:-1])
            return class_qualifier, method_name
        else: return None, parts[0] if parts else None
def get_fqn_parts_from_fd_all_round_logs_path(all_round_logs_path_str):
    if not isinstance(all_round_logs_path_str, str) or "/all_rounds/" not in all_round_logs_path_str:
        logger.debug(f"Invalid all_round_logs_path_str for FQN parsing: {all_round_logs_path_str}")
        return None, None
    try:
        normalized_path = all_round_logs_path_str.replace('\\', '/')
        path_segments = normalized_path.split('/')
        if len(path_segments) >= 2:
            fqn_method_segment = path_segments[-2]
            if '.' in fqn_method_segment:
                last_dot_index = fqn_method_segment.rfind('.')
                if last_dot_index != -1 and last_dot_index < len(fqn_method_segment) - 1 :
                    class_qualifier = fqn_method_segment[:last_dot_index]
                    method_name = fqn_method_segment[last_dot_index+1:]
                    if '.' in method_name:
                        logger.warning(f"Parsed method name '{method_name}' from '{fqn_method_segment}' contains a dot, might be incorrect.")
                        return class_qualifier, method_name
                    logger.debug(f"Parsed from '{fqn_method_segment}': ClassQ='{class_qualifier}', Method='{method_name}'")
                    return class_qualifier, method_name
                else:
                    logger.warning(f"Segment '{fqn_method_segment}' from all_round_logs after rsplit does not yield two parts for Class.method.")
                    return fqn_method_segment, None
            else:
                logger.warning(f"Segment '{fqn_method_segment}' from all_round_logs does not contain '.' separator.")
                return fqn_method_segment, None
        else:
            logger.warning(f"Not enough segments in all_round_logs_path: {all_round_logs_path_str}")
            return None, None
    except Exception as e:
        logger.error(f"Error parsing all_round_logs_path '{all_round_logs_path_str}': {e}", exc_info=True)
        return None, None
def get_java_file_path_and_actual_name(project_dir, module,
                                     class_qualifier_for_file_search,
                                     fallback_full_test_name_str, fallback_source_file_type):
    final_class_qualifier = class_qualifier_for_file_search
    if not final_class_qualifier:
        logger.debug(f"class_qualifier_for_file_search not provided, falling back to parsing {fallback_full_test_name_str}")
        final_class_qualifier, _ = get_fqn_parts_from_input_csv(fallback_full_test_name_str, fallback_source_file_type)
        if not final_class_qualifier:
            logger.warning(f"Could not extract class qualifier from fallback: {fallback_full_test_name_str}")
            return None, None
    package_parts = final_class_qualifier.split('.')[:-1] if '.' in final_class_qualifier else []
    simple_class_name_from_qualifier = final_class_qualifier.split('.')[-1]
    package_path = os.sep.join(package_parts)
    module_base_path = os.path.join(project_dir, module if module and module != "." else "")
    test_src_roots_patterns = [
        os.path.join(module_base_path, "src", "test", "java"), os.path.join(module_base_path, "src", "test"),
        os.path.join(module_base_path, "src", "main", "java"), os.path.join(module_base_path, "test", "java"),
        os.path.join(module_base_path, "test"), module_base_path
    ]
    for test_root in test_src_roots_patterns:
        if not os.path.isdir(test_root): continue
        potential_class_dir = os.path.join(test_root, package_path)
        if not os.path.isdir(potential_class_dir): continue
        expected_fname = simple_class_name_from_qualifier + ".java"
        path_try_direct = os.path.join(potential_class_dir, expected_fname)
        if os.path.exists(path_try_direct):
             return path_try_direct, expected_fname
        for f_name_in_dir in os.listdir(potential_class_dir):
            if f_name_in_dir.endswith(".java"):
                actual_simple_class_name_in_file = f_name_in_dir[:-5]
                if actual_simple_class_name_in_file.lower() == simple_class_name_from_qualifier.lower():
                    return os.path.join(potential_class_dir, f_name_in_dir), f_name_in_dir
    logger.warning(f"Robust Java file search failed for class qualifier '{final_class_qualifier}' in module '{module}' at '{project_dir}'")
    return None, None
def get_test_qualifier_from_full(full_test_name, source_file_type):
    if not isinstance(full_test_name, str): return ""
    parts = full_test_name.split('.')
    if source_file_type == "flaky_doctor_131_gpt_fixed_checked_labeled_time.csv":
        if len(parts) >= 2 and parts[-1].lower() == parts[-2].lower(): return ".".join(parts[:-1])
    return full_test_name
def normalize_fqn_for_matching(fqn_str):
    if not isinstance(fqn_str, str): return ""
    parts = fqn_str.split('.')
    if len(parts) < 2: return fqn_str.lower()
    method_name = parts[-1].lower(); class_name_part = parts[-2]; package_parts = parts[:-2]
    normalized_class_name = class_name_part.replace("_", "").lower()
    if package_parts: return ".".join(p.lower() for p in package_parts) + "." + normalized_class_name + "." + method_name
    else: return normalized_class_name + "." + method_name
def parse_patch_file_for_import_pom(patch_file_path):
    imports = []; pom = None
    try:
        with open(patch_file_path, 'r', encoding='utf-8', errors='replace') as f: content = f.read()
        last_round_match = None
        for match in re.finditer(r"ROUND\s*(\d+):", content, re.IGNORECASE): last_round_match = match
        if not last_round_match: logger.debug(f"No ROUND in {patch_file_path}"); return [], None
        round_content = content[last_round_match.start():]
        import_block_str = ""; m_after = re.search(r"After stitching:.*?import:\s*\n(.*?)(?:\n\s*(?:pom:|test_code:)|\Z)", round_content, re.DOTALL | re.IGNORECASE); m_before = re.search(r"Before stitching:.*?import:\s*\n(.*?)(?:\n\s*(?:pom:|test_code:)|\Z)", round_content, re.DOTALL | re.IGNORECASE)
        if m_after: import_block_str = m_after.group(1).strip()
        elif m_before: import_block_str = m_before.group(1).strip()
        if import_block_str and import_block_str.lower() not in ("[]", "none"):
            current_imports = []
            for line in import_block_str.splitlines():
                cl = line.strip().strip(",'\"")
                if cl.startswith("import ") and cl.endswith(";"): current_imports.append(cl)
                elif cl.startswith("['") and cl.endswith("']"):
                    try:
                        ev_list = eval(cl)
                        if isinstance(ev_list, list): current_imports.extend(i for i in ev_list if isinstance(i, str) and i.startswith("import ") and i.endswith(";"))
                    except: pass
            imports = current_imports
        pom_block_str = ""; m_pom_after = re.search(r"After stitching:.*?pom:\s*\n(.*?)(?:\n\s*test_code:|\Z)", round_content, re.DOTALL | re.IGNORECASE); m_pom_before = re.search(r"Before stitching:.*?pom:\s*\n(.*?)(?:\n\s*test_code:|\Z)", round_content, re.DOTALL | re.IGNORECASE)
        if m_pom_after: pom_block_str = m_pom_after.group(1).strip()
        elif m_pom_before: pom_block_str = m_pom_before.group(1).strip()
        if pom_block_str and pom_block_str.lower() != "none": pom = pom_block_str
    except FileNotFoundError: logger.error(f"Patch file not found: {patch_file_path}")
    except Exception as e: logger.error(f"Error parsing patch {patch_file_path}: {e}", exc_info=True)
    return imports, pom
def extract_method_body(method_code_str):
    if not isinstance(method_code_str, str): return None
    try:
        first_brace = method_code_str.index('{'); open_braces = 1 ; last_brace_index = -1
        for i, char in enumerate(method_code_str[first_brace + 1:]):
            actual_index = first_brace + 1 + i
            if char == '{': open_braces += 1
            elif char == '}':
                open_braces -= 1
                if open_braces == 0: last_brace_index = actual_index; break
        if last_brace_index != -1: return method_code_str[first_brace + 1 : last_brace_index].strip()
    except ValueError: logger.warning(f"Opening brace '{{' not found for body extraction.")
    except Exception as e: logger.warning(f"Error in extract_method_body: {e}")
    logger.warning(f"Could not extract body reliably from method code string. Content: {method_code_str[:100]}...")
    return None
def apply_code_patch(java_file_path, original_java_content, patched_method_code_from_csv,
                     method_name_to_find, source_file_type_for_annotation_heuristic):
    logger.info(f"Applying patch to {java_file_path} for method '{method_name_to_find}'")
    if not method_name_to_find: logger.error("Method name to find is None, cannot apply patch."); return False
    if not original_java_content: logger.error(f"Original Java content for {java_file_path} is empty. Cannot apply patch."); return False
    original_method_info = fd_utils.extract_method(method_name_to_find, original_java_content)
    if not original_method_info or not original_method_info[0] or not original_method_info[1]:
        logger.error(f"FD_UTILS: Could not extract original method '{method_name_to_find}' from {java_file_path}. Patch application failed.")
        return False
    original_method_full_code_from_fd = original_method_info[0]
    original_method_node = original_method_info[1]
    original_declaration_head = original_method_full_code_from_fd.split("{", 1)[0].strip() if "{" in original_method_full_code_from_fd else original_method_full_code_from_fd.strip()
    has_test_in_original_decl = bool(re.search(r"@Test(?:\s*\(.*?\))?|@org\.junit\.Test(?:\s*\(.*?\))?|@junit\.framework\.Test(?:\s*\(.*?\))?", original_declaration_head, re.IGNORECASE))
    has_test_in_ast = False
    if original_method_node and original_method_node.annotations:
        for ann in original_method_node.annotations:
            if ann.name.lower() == "test" or ann.name.lower().endswith(".test"): has_test_in_ast = True; break
    final_declaration_to_use = original_declaration_head
    if not has_test_in_original_decl and not has_test_in_ast and method_name_to_find.lower().startswith("test"):
        logger.info(f"Method '{method_name_to_find}' seems to be a test but lacks @Test. Prepending @org.junit.Test.")
        lines_decl = final_declaration_to_use.splitlines()
        if lines_decl:
            first_line_indent = len(lines_decl[0]) - len(lines_decl[0].lstrip())
            indent_str = " " * first_line_indent
            if lines_decl[0].strip().startswith("@"): final_declaration_to_use = indent_str + "@org.junit.Test\n" + final_declaration_to_use
            else: final_declaration_to_use = indent_str + "@org.junit.Test\n" + indent_str + final_declaration_to_use.lstrip()
        else: final_declaration_to_use = "@org.junit.Test\n" + final_declaration_to_use
    patched_body_content = extract_method_body(patched_method_code_from_csv)
    if patched_body_content is None:
        logger.error(f"Failed to extract method body from patched_code for {method_name_to_find}. Cannot apply patch.")
        return False
    if not final_declaration_to_use.endswith("{"): final_declaration_to_use = final_declaration_to_use.strip() + " {"
    else: final_declaration_to_use = final_declaration_to_use.strip()
    indented_patched_body = "";
    if patched_body_content.strip():
        body_lines = patched_body_content.splitlines()
        base_indent_match = re.search(r"\n(\s+)\S", original_method_full_code_from_fd.split("{\n", 1)[1] if "{\n" in original_method_full_code_from_fd else "")
        base_indent = base_indent_match.group(1) if base_indent_match else "    "
        indented_patched_body = "\n".join([base_indent + line.strip() for line in body_lines if line.strip()])
        if indented_patched_body and not indented_patched_body.startswith(base_indent) and body_lines:
             indented_patched_body = base_indent + body_lines[0] + ("\n" + "\n".join([base_indent + bl for bl in body_lines[1:]]) if len(body_lines) > 1 else "")
    closing_brace_indent = ""; original_lines = original_method_full_code_from_fd.splitlines()
    if original_lines and original_lines[-1].strip() == "}": closing_brace_indent = original_lines[-1][:original_lines[-1].find("}")]
    new_full_method_code = final_declaration_to_use + "\n" + indented_patched_body + "\n" + closing_brace_indent + "}\n"
    try:
        if original_method_full_code_from_fd not in original_java_content:
            logger.error(f"The exact original method code (from fd_utils for '{method_name_to_find}') was not found in {java_file_path}.")
            return False
        new_content = original_java_content.replace(original_method_full_code_from_fd, new_full_method_code, 1)
        if new_content == original_java_content:
             logger.error(f"Code replacement did not change content for {java_file_path}."); return False
        with open(java_file_path, 'w', encoding='utf-8', errors='replace') as f: f.write(new_content)
        logger.info(f"Applied code patch to {java_file_path} for '{method_name_to_find}'.")
        return True
    except Exception as e: logger.error(f"Error writing patched content to {java_file_path}: {e}", exc_info=True); return False
def add_imports_to_java_file(java_file_path, imports_to_add):
    if not imports_to_add: return True
    try:
        with open(java_file_path, 'r', encoding='utf-8', errors='replace') as f: lines = f.readlines()
        existing_imports = set(); package_line_index = -1; last_import_line_index = -1; first_class_or_interface_line_index = -1
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith("package "): package_line_index = i
            elif stripped_line.startswith("import "): existing_imports.add(stripped_line); last_import_line_index = i
            elif first_class_or_interface_line_index == -1 and \
                 (re.match(r"^(public|protected|private|class|interface|enum|@)", stripped_line) and not stripped_line.startswith("import")):
                first_class_or_interface_line_index = i
        new_imports_to_insert_str = ""
        for imp in imports_to_add:
            clean_imp = str(imp).strip()
            if clean_imp.startswith("import ") and clean_imp.endswith(";") and clean_imp not in existing_imports:
                new_imports_to_insert_str += clean_imp + "\n"; existing_imports.add(clean_imp)
        if new_imports_to_insert_str:
            insert_index = 0
            if last_import_line_index != -1: insert_index = last_import_line_index + 1
            elif package_line_index != -1: insert_index = package_line_index + 1; new_imports_to_insert_str = "\n" + new_imports_to_insert_str
            elif first_class_or_interface_line_index != -1: insert_index = first_class_or_interface_line_index; new_imports_to_insert_str = new_imports_to_insert_str + "\n"
            lines.insert(insert_index, new_imports_to_insert_str)
            with open(java_file_path, 'w', encoding='utf-8', errors='replace') as f: f.writelines(lines)
            logger.info(f"Added imports to {java_file_path}: {new_imports_to_insert_str.strip().splitlines()}")
        else: logger.info(f"No new imports to add to {java_file_path}.")
        return True
    except Exception as e: logger.error(f"Error adding imports to {java_file_path}: {e}"); return False
def safe_add_dependencies_to_pom(pom_file_path, pom_dependencies_str):
    if not pom_dependencies_str or not pom_dependencies_str.strip() or pom_dependencies_str.lower() == "none":
        logger.debug(f"No POM dependencies to add for {pom_file_path}."); return True
    cleaned_deps = pom_dependencies_str.strip()
    cleaned_deps = re.sub(r"<!--\s*<pom.xml start>\s*-->", "", cleaned_deps, flags=re.IGNORECASE).strip()
    cleaned_deps = re.sub(r"<!--\s*<pom.xml end>\s*-->", "", cleaned_deps, flags=re.IGNORECASE).strip()
    cleaned_deps = re.sub(r"<!--.*?-->", "", cleaned_deps, flags=re.DOTALL).strip()
    if cleaned_deps.lower().startswith("<dependencies>") and cleaned_deps.lower().endswith("</dependencies>"):
        match_inner = re.search(r"<dependencies>([\s\S]*)</dependencies>", cleaned_deps, re.IGNORECASE | re.DOTALL)
        if match_inner: cleaned_deps = match_inner.group(1).strip()
    if not cleaned_deps: logger.info(f"POM dependencies string was empty for {pom_file_path}."); return True
    try:
        logger.info(f"Attempting to add dependencies to {pom_file_path} using fd_add_dependency.")
        fd_add_dependency(pom_file_path, cleaned_deps)
        logger.info(f"Successfully called fd_add_dependency for {pom_file_path}.")
        return True
    except Exception as e:
        logger.error(f"Error using fd_add_dependency for {pom_file_path}: {e}. Falling back.", exc_info=True)
        try:
            with open(pom_file_path, 'r', encoding='utf-8', errors='replace') as f: content = f.read()
            dependencies_tag_match = re.search(r"<dependencies>", content, re.IGNORECASE)
            if dependencies_tag_match:
                insert_point = dependencies_tag_match.end(); new_content = content[:insert_point] + "\n" + cleaned_deps + "\n" + content[insert_point:]
            else:
                project_tag_match = re.search(r"</project>", content, re.IGNORECASE)
                if project_tag_match:
                    insert_point = project_tag_match.start(); deps_to_insert = f"\n  <dependencies>\n    {cleaned_deps}\n  </dependencies>\n"
                    new_content = content[:insert_point] + deps_to_insert + content[insert_point:]
                else: logger.error(f"Fallback POM: <dependencies> and </project> tag not found."); return False
            with open(pom_file_path, 'w', encoding='utf-8', errors='replace') as f: f.write(new_content)
            logger.info(f"Fallback POM: Successfully modified {pom_file_path}.")
            return True
        except Exception as e_fallback: logger.error(f"Fallback POM: Error adding: {e_fallback}"); return False
def run_maven_command(project_dir, command, module=None):
    cwd = os.getcwd()
    try:
        if not os.path.isdir(project_dir):
            logger.error(f"Project directory '{project_dir}' does not exist for maven command.")
            return False, "ProjectDirNotFound", f"Project directory '{project_dir}' does not exist."
        os.chdir(project_dir)
        full_command = ["mvn"] + command
        effective_module = module if module and module != "." else None
        if effective_module and "-pl" not in command:
            full_command.extend(["-pl", effective_module, "-am"])
        logger.info(f"Running Maven command: {' '.join(full_command)} in {project_dir}")
        process = subprocess.run(full_command, capture_output=True, text=True, check=False, timeout=1200, errors='replace')
        build_successful = "BUILD SUCCESS" in process.stdout
        if process.returncode == 0 and build_successful:
            logger.info(f"Maven command successful: {' '.join(full_command)}")
            return True, process.stdout, process.stderr
        else:
            logger.error(f"Maven command failed: {' '.join(full_command)} (Return Code: {process.returncode})")
            error_log_lines = []; compilation_error_block = False
            combined_output_lines = (process.stdout + "\n" + process.stderr).splitlines()
            for line_idx, line in enumerate(combined_output_lines):
                if "[ERROR] COMPILATION ERROR" in line: compilation_error_block = True
                elif compilation_error_block and line.startswith("[ERROR]"): error_log_lines.append(line)
                elif compilation_error_block and not line.startswith("["): error_log_lines.append(line)
                elif "BUILD FAILURE" in line: compilation_error_block = True; error_log_lines.append(line)
                elif "Failed to execute goal" in line:
                    if line not in error_log_lines : error_log_lines.append(line)
                    try:
                        all_lines = combined_output_lines; idx_find = -1
                        for current_idx, l_find in enumerate(all_lines):
                            if l_find == line: idx_find = current_idx; break
                        if idx_find != -1:
                            for next_line_idx in range(idx_find + 1, min(idx_find + 6, len(all_lines))):
                                if all_lines[next_line_idx] not in error_log_lines: error_log_lines.append(all_lines[next_line_idx])
                    except ValueError: pass
            if error_log_lines: logger.error("Relevant Build/Compilation Errors:\n" + "\n".join(error_log_lines[:50]))
            else:
                 logger.error(f"Stdout (last 1000 chars): {process.stdout[-1000:]}")
                 logger.error(f"Stderr (last 1000 chars): {process.stderr[-1000:]}")
            return False, process.stdout, process.stderr
    except subprocess.TimeoutExpired: logger.error(f"Maven command timed out: {' '.join(full_command)}"); return False, "Timeout", "Timeout"
    except Exception as e: logger.error(f"Exception during Maven command {' '.join(full_command)}: {e}", exc_info=True); return False, str(e), str(e)
    finally:
        if os.getcwd() != cwd :
            try: os.chdir(cwd)
            except FileNotFoundError: logger.error(f"Critical error: Could not chdir back to original cwd '{cwd}'. Current dir: {os.getcwd()}")
def run_nondex_test(project_dir, module, nondex_test_name_param, jdk_version="8", nondex_internal_runs_count_str="1"):
    logger.info(f"Running NonDex for: {nondex_test_name_param} in module '{module if module and module != '.' else 'root'}' with JDK {jdk_version} (NonDex plugin runs: {nondex_internal_runs_count_str})")
    if not os.path.exists(FD_RUN_NONDEX_CMD):
        logger.error(f"NonDex script not found at {FD_RUN_NONDEX_CMD}");
        return "NONDEX_SCRIPT_MISSING", [f"NONDEX_SCRIPT_MISSING"] * int(nondex_internal_runs_count_str)
    abs_project_dir = os.path.abspath(project_dir)
    effective_module_for_script = module if module and module != "." else ""
    try:
        command_parts = ["bash", FD_RUN_NONDEX_CMD, abs_project_dir, effective_module_for_script, nondex_test_name_param, jdk_version, str(nondex_internal_runs_count_str)]
        logger.info(f"Shell command for NonDex: {' '.join(command_parts)}")
        process = subprocess.run(
            command_parts,
            capture_output=True, text=True, check=False, timeout=900, errors='replace'
        )
        output = process.stdout + "\n" + process.stderr
        logger.info(f"--- Full NonDex output for {nondex_test_name_param} (PID: {process.pid if hasattr(process, 'pid') else 'N/A'}) ---")
        output_lines_for_log = output.splitlines()
        if len(output_lines_for_log) > 200:
            for line in output_lines_for_log[:100]: logger.info(f"[NonDex Output] {line}")
            logger.info("[NonDex Output] ... (output truncated) ...")
            for line in output_lines_for_log[-100:]: logger.info(f"[NonDex Output] {line}")
        else:
            for line in output_lines_for_log: logger.info(f"[NonDex Output] {line}")
        logger.info(f"--- End NonDex Output for {nondex_test_name_param} ---")
        if "Missing argument for option: pl" in output or "Unknown lifecycle phase \".\"" in output :
             logger.error(f"NonDex execution failed due to Maven argument error (likely -pl issue with module='{effective_module_for_script}').")
             return "MAVEN_ARG_ERROR", ["Maven Arg Error"] * int(nondex_internal_runs_count_str)
        if "COMPILATION ERROR" in output: return "COMPILATION_ERROR", ["Compilation Error"] * int(nondex_internal_runs_count_str)
        if "BUILD FAILURE" in output and "Tests run:" not in output and "NonDex SUMMARY" not in output :
            logger.warning(f"NonDex task for {nondex_test_name_param} resulted in BUILD FAILURE before test execution.")
            return "BUILD_FAILURE", ["Build Failure (No Tests/Summary)"] * int(nondex_internal_runs_count_str)
        if "No tests were executed!" in output or \
           ("No tests found for given includes" in output and "NonDex SUMMARY" not in output) :
            logger.warning(f"NonDex reported no tests executed for {nondex_test_name_param}.")
            return "NO_TESTS_EXECUTED", ["No Tests Executed By NonDex"] * int(nondex_internal_runs_count_str)
        target_class_name_for_log_check = nondex_test_name_param.split('#')[0]
        capture_summary_line_pattern = re.compile(
            r"^(?:\[INFO\]|\[ERROR\])?\s*(Tests run:\s*1,\s*Failures:\s*\d+,\s*Errors:\s*\d+,\s*Skipped:\s*\d+.*?(?:in\s+|--\s+in\s+)" +
            re.escape(target_class_name_for_log_check) + r".*?(?:<<< FAILURE!)?)$", re.MULTILINE
        )
        simple_capture_summary_line_pattern = re.compile(
             r"^(?:\[INFO\]|\[ERROR\])?\s*(Tests run:\s*1,\s*Failures:\s*\d+,\s*Errors:\s*\d+,\s*Skipped:\s*\d+.*?)$", re.MULTILINE
        )
        all_individual_run_details = capture_summary_line_pattern.findall(output)
        if not all_individual_run_details:
            logger.debug(f"Specific summary line regex found no matches for {target_class_name_for_log_check}. Trying generic pattern.")
            all_individual_run_details = simple_capture_summary_line_pattern.findall(output)
        logger.debug(f"Found {len(all_individual_run_details)} potential test summary lines for {nondex_test_name_param}: {all_individual_run_details}")
        num_expected_runs = int(nondex_internal_runs_count_str)
        if len(all_individual_run_details) >= num_expected_runs:
            final_details_to_process = all_individual_run_details[-num_expected_runs:]
            logger.info(f"Collected {len(final_details_to_process)} NonDex run details for {nondex_test_name_param}.")
            return "COLLECTED_RUNS", final_details_to_process
        elif all_individual_run_details:
             logger.warning(f"Expected {num_expected_runs} NonDex run summaries for {nondex_test_name_param}, but found only {len(all_individual_run_details)}. Using found details and filling missing.")
             collected_details_list = [line.strip() for line in all_individual_run_details]
             collected_details_list.extend(
                 ["MISSING_NONDEX_INTERNAL_RUN_SUMMARY"] * (num_expected_runs - len(all_individual_run_details))
             )
             return "COLLECTED_RUNS_INCOMPLETE", collected_details_list
        else:
            if "NonDex: The test passed in all configurations" in output or \
               ("[INFO] All tests pass without NonDex shuffling" in output and "BUILD SUCCESS" in output):
                 return "TEST_PASS_OVERALL", ["NonDex summary: Passed (no individual run line found)"] * num_expected_runs
            if "NonDex: The test is flaky" in output or "Nondex found a flakiness" in output or \
               (re.search(r"\[WARNING\]\s*" + re.escape(nondex_test_name_param), output) and ("BUILD FAILURE" in output or "Failures: 1" in output or "Errors: 1" in output)):
                return "TEST_FAILURE_OVERALL", ["NonDex summary: Flaky (no individual run line found)"] * num_expected_runs
            if "No tests were executed!" in output or "No tests found for given includes" in output:
                logger.warning(f"NonDex reported no tests executed for {nondex_test_name_param} (final summary check).")
                return "NO_TESTS_EXECUTED", ["No Tests Executed By NonDex (summary)"] * num_expected_runs
            if "BUILD FAILURE" in output:
                 logger.warning(f"NonDex task resulted in BUILD FAILURE for {nondex_test_name_param} and no specific test failure line found.")
                 return "BUILD_FAILURE", ["NonDex BUILD FAILURE (no specific test line)"] * num_expected_runs
            logger.warning(f"No clear test result indicators in NonDex output for {nondex_test_name_param}.")
            return "UNKNOWN_NONDEX_RESULT", [None] * num_expected_runs
    except subprocess.TimeoutExpired: logger.error(f"NonDex run timed out for {nondex_test_name_param}"); return "TIMEOUT_ERROR", [None] * int(nondex_internal_runs_count_str)
    except Exception as e: logger.error(f"Exception during NonDex run for {nondex_test_name_param}: {e}", exc_info=True); return "EXECUTION_ERROR", [None] * int(nondex_internal_runs_count_str)
def find_pom_xml(start_dir, module_name):
    if not start_dir or not os.path.isdir(start_dir): logger.warning(f"Invalid start_dir for find_pom_xml: {start_dir}"); return None
    if module_name == ".": module_path = start_dir
    else: module_path = os.path.join(start_dir, module_name)
    pom_in_module = os.path.join(module_path, "pom.xml")
    if os.path.exists(pom_in_module): return pom_in_module
    pom_in_root = os.path.join(start_dir, "pom.xml")
    if os.path.exists(pom_in_root): logger.debug(f"POM not found in module '{module_name}', using root POM: {pom_in_root}"); return pom_in_root
    logger.warning(f"Could not find pom.xml in module '{module_name}' or project root '{start_dir}'"); return None
def attempt_import_stitching(java_file_path, project_dir, module, full_test_name_for_stitch, jdk_version,
                             original_java_content_for_stitching, initial_build_stdout, initial_build_stderr):
    logger.info(f"Attempting import stitching for {java_file_path}...")
    current_java_content = original_java_content_for_stitching
    added_imports_during_stitching = []
    build_output_for_parsing = initial_build_stdout + "\n" + initial_build_stderr
    symbol_class_pattern = r"符号:\s*(?:类|接口|枚举)\s*([\w\.<>]+)"
    package_not_exist_pattern = r"程序包\s*([\w\.]+)\s*不存在"
    missing_symbols_from_log = set(re.findall(symbol_class_pattern, build_output_for_parsing))
    missing_packages_from_log = set(re.findall(package_not_exist_pattern, build_output_for_parsing))
    if not missing_symbols_from_log and not missing_packages_from_log :
        logger.info("No 'symbol: class/interface/enum X' or 'package X does not exist' errors found for stitching.")
        if "COMPILATION ERROR" not in build_output_for_parsing and "BUILD FAILURE" not in build_output_for_parsing:
            logger.info("Initial build failure was not a compilation error. Stitching might not help.")
            return False, current_java_content, added_imports_during_stitching
    logger.info(f"Potential missing symbols: {missing_symbols_from_log}, missing packages: {missing_packages_from_log}")
    max_stitching_attempts = 5
    for attempt in range(max_stitching_attempts):
        made_change_in_this_attempt = False
        symbols_to_retry_this_round = list(missing_symbols_from_log)
        missing_symbols_from_log.clear()
        if not symbols_to_retry_this_round:
            logger.info(f"Stitching attempt {attempt+1}: No more symbols to fix this round based on previous errors.")
            final_check_build_cmd = ["clean", "install", "-DskipTests", "-Dmaven.javadoc.skip=true", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dfindbugs.skip=true", "-Denforcer.skip=true", "-Dspotbugs.skip=true", "-Djacoco.skip=true", "-Dspotless.check.skip=true", "-Dmaven.test.failure.ignore=true", "-DfailIfNoTests=false"]
            build_ok_after_round, _, _ = run_maven_command(project_dir, final_check_build_cmd, module)
            if build_ok_after_round: logger.info("Build became successful during stitching iterations."); return True, current_java_content, added_imports_during_stitching
            break
        logger.info(f"Stitching attempt {attempt+1} for symbols: {symbols_to_retry_this_round}")
        for tofix_symbol_full_qualifier in symbols_to_retry_this_round:
            tofix_symbol_simple = tofix_symbol_full_qualifier.split('.')[-1]
            if tofix_symbol_simple not in java_standard_libs_data:
                logger.debug(f"Symbol '{tofix_symbol_simple}' not in java_standard_libs.json.")
                missing_symbols_from_log.add(tofix_symbol_full_qualifier); continue
            found_solving_import = False
            for potential_import_statement_list in java_standard_libs_data[tofix_symbol_simple]:
                potential_import = potential_import_statement_list[0].strip() if isinstance(potential_import_statement_list, list) and potential_import_statement_list else \
                                   potential_import_statement_list.strip() if isinstance(potential_import_statement_list, str) else None
                if not potential_import: continue
                logger.info(f"Trying to add import '{potential_import}' for symbol '{tofix_symbol_simple}'")
                temp_java_lines = current_java_content.splitlines(True)
                temp_existing_imports = set(); temp_package_idx, temp_last_import_idx, temp_first_class_idx = -1, -1, -1
                for k, l_line in enumerate(temp_java_lines):
                    sl = l_line.strip()
                    if sl.startswith("package "): temp_package_idx = k
                    elif sl.startswith("import "): temp_existing_imports.add(sl); temp_last_import_idx = k
                    elif temp_first_class_idx == -1 and (re.match(r"^(public|protected|private|class|interface|enum|@)", sl) and not sl.startswith("import")):
                        temp_first_class_idx = k
                temp_stitched_content = current_java_content
                if potential_import not in temp_existing_imports:
                    insert_idx = 0; prefix = ""; suffix = "\n"
                    if temp_last_import_idx != -1: insert_idx = temp_last_import_idx + 1
                    elif temp_package_idx != -1: insert_idx = temp_package_idx + 1; prefix = "\n"
                    elif temp_first_class_idx != -1: insert_idx = temp_first_class_idx; suffix = "\n\n"
                    temp_java_lines.insert(insert_idx, prefix + potential_import + suffix)
                    temp_stitched_content = "".join(temp_java_lines)
                with open(java_file_path, 'w', encoding='utf-8', errors='replace') as f_temp: f_temp.write(temp_stitched_content)
                logger.info(f"Rebuilding after adding import '{potential_import}'...")
                stitch_build_command = ["clean", "install", "-DskipTests", "-Dmaven.compiler.failOnError=false", "-Dmaven.javadoc.skip=true", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dfindbugs.skip=true", "-Denforcer.skip=true", "-Dspotbugs.skip=true", "-Djacoco.skip=true", "-Dspotless.check.skip=true", "-Dmaven.test.failure.ignore=true", "-DfailIfNoTests=false"]
                build_ok, out, err = run_maven_command(project_dir, stitch_build_command, module)
                current_build_output_after_import = out + "\n" + err
                errors_for_this_symbol_after_import = [
                    msg_line for msg_line in current_build_output_after_import.splitlines()
                    if (f"符号:   类 {tofix_symbol_simple}" in msg_line or f"符号: 类 {tofix_symbol_simple}" in msg_line or f"符号: 接口 {tofix_symbol_simple}" in msg_line or f"符号: 枚举 {tofix_symbol_simple}" in msg_line)
                       and "找不到符号" in current_build_output_after_import
                ]
                if not errors_for_this_symbol_after_import:
                    logger.info(f"Import '{potential_import}' likely fixed symbol '{tofix_symbol_simple}'.")
                    current_java_content = temp_stitched_content
                    if potential_import not in added_imports_during_stitching : added_imports_during_stitching.append(potential_import)
                    made_change_in_this_attempt = True; found_solving_import = True
                    missing_symbols_from_log.update(set(re.findall(symbol_class_pattern, current_build_output_after_import)))
                    missing_symbols_from_log.update(set(re.findall(package_not_exist_pattern, current_build_output_after_import)))
                    break
                else:
                    logger.debug(f"Import '{potential_import}' did not fix symbol '{tofix_symbol_simple}'.")
                    with open(java_file_path, 'w', encoding='utf-8', errors='replace') as f_restore_iter:
                        f_restore_iter.write(current_java_content)
            if not found_solving_import:
                logger.warning(f"Could not fix symbol '{tofix_symbol_simple}' with available standard imports.")
                missing_symbols_from_log.add(tofix_symbol_full_qualifier)
        if not made_change_in_this_attempt and missing_symbols_from_log:
             logger.warning(f"Stitching attempt {attempt+1} made no progress. Remaining: {missing_symbols_from_log}. Stopping."); break
        if not missing_symbols_from_log:
            logger.info("All identified symbols during this stitching iteration seem to be resolved or not in standard libs.")
            final_build_ok_inner, _, _ = run_maven_command(project_dir, ["clean", "install", "-DskipTests", "-Dmaven.javadoc.skip=true", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dfindbugs.skip=true", "-Denforcer.skip=true", "-Dspotbugs.skip=true", "-Djacoco.skip=true", "-Dspotless.check.skip=true", "-Dmaven.test.failure.ignore=true", "-DfailIfNoTests=false"], module)
            if final_build_ok_inner:
                logger.info("Build successful after current stitching iteration.")
                return True, current_java_content, added_imports_during_stitching
            else:
                logger.warning("Build still failing after resolving symbols in this iteration. Other errors might exist.")
                _, new_out, new_err = run_maven_command(project_dir, ["clean", "install", "-DskipTests", "-Dmaven.compiler.failOnError=false", "-Dmaven.javadoc.skip=true", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dfindbugs.skip=true", "-Denforcer.skip=true", "-Dspotbugs.skip=true", "-Djacoco.skip=true", "-Dspotless.check.skip=true", "-Dmaven.test.failure.ignore=true", "-DfailIfNoTests=false"], module)
                missing_symbols_from_log.update(set(re.findall(symbol_class_pattern, new_out + "\n" + new_err)))
                if not missing_symbols_from_log: break
    final_build_ok_after_loop, _, _ = run_maven_command(project_dir, ["clean", "install", "-DskipTests", "-Dmaven.javadoc.skip=true", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dfindbugs.skip=true", "-Denforcer.skip=true", "-Dspotbugs.skip=true", "-Djacoco.skip=true", "-Dspotless.check.skip=true", "-Dmaven.test.failure.ignore=true", "-DfailIfNoTests=false"], module)
    if final_build_ok_after_loop: logger.info("Final build check after stitching loop is SUCCESSFUL.")
    else: logger.warning(f"Final build check after stitching loop FAILED. Unresolved symbols: {missing_symbols_from_log}")
    return final_build_ok_after_loop, current_java_content, added_imports_during_stitching
def preprocess_data():
    logger.info("Starting Stage 1: Preprocessing data to extract imports, POMs, and actual test name components.")
    if not os.path.exists(MERGED_DATASET_CSV):
        logger.error(f"Input file {MERGED_DATASET_CSV} not found."); return
    fd_results_cache = {}
    processed_rows = []
    with open(MERGED_DATASET_CSV, 'r', encoding='utf-8', errors='replace') as infile:
        reader = csv.DictReader(infile)
        original_fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        fieldnames = original_fieldnames[:]
        for col in ['fd_import', 'fd_pom', 'fd_actual_class_qualifier', 'fd_actual_method_name']:
            if col not in fieldnames: fieldnames.append(col)
        for i, row in enumerate(reader):
            project_name_csv = row['project_name']
            full_test_name_from_input = row['full_test_name']
            source_file = row['source_file']
            repo_url = row['repo_url']
            original_sha = row['original_sha']
            row['fd_import'] = "[]"; row['fd_pom'] = "None"
            row['fd_actual_class_qualifier'] = None
            row['fd_actual_method_name'] = None
            if source_file == "laky_fix_181_gpt35_labeled.csv":
                processed_rows.append(row); continue
            patch_main_dir_name = ""; fd_results_paths_bases = []
            if source_file == "flaky_doctor_126_magicoder_with_time.csv":
                patch_main_dir_name = "magicoder"; fd_results_paths_bases.append(os.path.join(PATCH_BASE_DIR, patch_main_dir_name))
            elif source_file == "flaky_doctor_131_gpt_fixed_checked_labeled_time.csv":
                patch_main_dir_name = "gpt";
                fd_results_paths_bases.append(os.path.join(PATCH_BASE_DIR, patch_main_dir_name, "gpt1"))
                fd_results_paths_bases.append(os.path.join(PATCH_BASE_DIR, patch_main_dir_name, "gpt2"))
            else:
                logger.warning(f"Unknown source_file type: {source_file} for row {i+1}. Skipping."); processed_rows.append(row); continue
            found_match_in_fd_csv = False
            for fd_result_base_path in fd_results_paths_bases:
                if found_match_in_fd_csv: break
                fd_results_csv_path_key = fd_result_base_path
                if fd_results_csv_path_key not in fd_results_cache:
                    actual_fd_csv = find_flakydoctor_results_csv(fd_result_base_path)
                    if not actual_fd_csv: fd_results_cache[fd_results_csv_path_key] = None; continue
                    try:
                        df = pd.read_csv(actual_fd_csv)
                        df['project_lower'] = df['project'].astype(str).str.lower()
                        df['sha_lower'] = df['sha'].astype(str).str.lower()
                        df['test_normalized_for_match'] = df['test'].astype(str).apply(normalize_fqn_for_matching)
                        fd_results_cache[fd_results_csv_path_key] = df
                    except Exception as e:
                        logger.error(f"Error reading/preprocessing FD CSV {actual_fd_csv}: {e}", exc_info=True)
                        fd_results_cache[fd_results_csv_path_key] = None; continue
                fd_df = fd_results_cache[fd_results_csv_path_key]
                if fd_df is None: continue
                target_repo_url_lower = str(repo_url).lower()
                target_original_sha_lower = str(original_sha).lower()
                standardized_input_test_name = get_test_qualifier_from_full(full_test_name_from_input, source_file)
                normalized_input_test_name_for_match = normalize_fqn_for_matching(standardized_input_test_name)
                match_condition = (
                    (fd_df['project_lower'] == target_repo_url_lower) &
                    (fd_df['sha_lower'] == target_original_sha_lower) &
                    (fd_df['test_normalized_for_match'] == normalized_input_test_name_for_match)
                )
                matched_fd_rows = fd_df[match_condition]
                if not matched_fd_rows.empty:
                    fd_row_original_case = fd_df.loc[matched_fd_rows.index[0]]
                    all_round_logs_path_suffix = fd_row_original_case.get('all_round_logs')
                    fd_class_qual, fd_method_name = get_fqn_parts_from_fd_all_round_logs_path(str(all_round_logs_path_suffix))
                    if fd_class_qual: row['fd_actual_class_qualifier'] = fd_class_qual
                    if fd_method_name: row['fd_actual_method_name'] = fd_method_name
                    if pd.notna(all_round_logs_path_suffix) and isinstance(all_round_logs_path_suffix, str) and "/all_rounds/" in all_round_logs_path_suffix:
                        path_parts = all_round_logs_path_suffix.split('/')
                        try:
                            all_rounds_index = path_parts.index("all_rounds")
                            relative_to_all_rounds_dir = os.path.join(*path_parts[all_rounds_index:])
                            absolute_patch_file = os.path.join(fd_result_base_path, relative_to_all_rounds_dir)
                            absolute_patch_file = os.path.normpath(absolute_patch_file)
                            imports, pom_str = parse_patch_file_for_import_pom(absolute_patch_file)
                            row['fd_import'] = str(imports) if imports else "[]"; row['fd_pom'] = pom_str if pom_str else "None"
                        except ValueError: logger.warning(f"'all_rounds' keyword not found in log path: {all_round_logs_path_suffix} for row {i+1}")
                        except Exception as e_path: logger.error(f"Error constructing patch path from '{all_round_logs_path_suffix}': {e_path}", exc_info=True)
                    found_match_in_fd_csv = True; break
            if not found_match_in_fd_csv:
                logger.warning(f"No matching FD entry for row {i+1}: URL={repo_url}, SHA={original_sha}, InputFQN='{full_test_name_from_input}', StandardizedFQN='{standardized_input_test_name}', NormalizedFQNForMatch='{normalized_input_test_name_for_match}'")
            processed_rows.append(row)
            if (i + 1) % 100 == 0: logger.info(f"Preprocessed {i+1} rows...")
    try:
        with open(MERGED_IMPORT_POM_CSV, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames); writer.writeheader(); writer.writerows(processed_rows)
        logger.info(f"Successfully created {MERGED_IMPORT_POM_CSV}.")
    except IOError as e: logger.error(f"Error writing {MERGED_IMPORT_POM_CSV}: {e}")
    logger.info("Stage 1: Preprocessing data finished.")
def apply_and_validate():
    logger.info("Starting Stage 2: Applying patches and running NonDex validation.")
    if not os.path.exists(MERGED_IMPORT_POM_CSV):
        logger.error(f"Input file {MERGED_IMPORT_POM_CSV} not found."); return
    Path(PATCHED_JAVA_FILES_DIR).mkdir(parents=True, exist_ok=True)
    Path(DEBUG_JAVA_FILES_DIR).mkdir(parents=True, exist_ok=True)
    processed_identifiers = set()
    if os.path.exists(NONDEX_VALIDATION_RESULTS_CSV):
        try:
            df_existing_results = pd.read_csv(NONDEX_VALIDATION_RESULTS_CSV, low_memory=False)
            for _, r_row in df_existing_results.iterrows():
                identifier = (
                    str(r_row.get('project_name')), str(r_row.get('full_test_name')),
                    str(r_row.get('original_sha')), str(r_row.get('generated_patch', ''))[:100],
                    str(r_row.get('source_file'))
                )
                if not any(val is None or (isinstance(val, float) and pd.isna(val)) for val in identifier):
                    processed_identifiers.add(identifier)
            logger.info(f"Loaded {len(processed_identifiers)} already processed entries from {NONDEX_VALIDATION_RESULTS_CSV}.")
        except Exception as e: logger.warning(f"Could not read existing results from {NONDEX_VALIDATION_RESULTS_CSV}: {e}.")
    with open(MERGED_IMPORT_POM_CSV, 'r', encoding='utf-8', errors='replace') as infile:
        reader = csv.DictReader(infile)
        original_fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        new_fieldnames = original_fieldnames[:]
        if 'actual_java_filename' not in new_fieldnames: new_fieldnames.append('actual_java_filename')
        if 'fd_actual_class_qualifier' not in new_fieldnames and 'fd_actual_class_qualifier' in original_fieldnames : new_fieldnames.append('fd_actual_class_qualifier')
        if 'fd_actual_method_name' not in new_fieldnames and 'fd_actual_method_name' in original_fieldnames : new_fieldnames.append('fd_actual_method_name')
        for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT):
            col_name = f'nondex_round{k_round_idx+1}'
            if col_name not in new_fieldnames : new_fieldnames.append(col_name)
        if 'nondex_consistency' not in new_fieldnames: new_fieldnames.append('nondex_consistency')
        write_header = not os.path.exists(NONDEX_VALIDATION_RESULTS_CSV) or os.path.getsize(NONDEX_VALIDATION_RESULTS_CSV) == 0
        with open(NONDEX_VALIDATION_RESULTS_CSV, 'a', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=new_fieldnames, extrasaction='ignore')
            if write_header: writer.writeheader()
            for i, row in enumerate(reader):
                logger.info(f"--- Processing row {i+1} from {MERGED_IMPORT_POM_CSV} for patch application and NonDex ---")
                project_name_in_csv = row['project_name']; repo_url = row['repo_url']; full_test_name_input = row['full_test_name']
                source_file_csv_type = row['source_file']; module = row['module']; original_sha_from_csv = str(row['original_sha']).strip()
                generated_patch_code = row['generated_patch']
                actual_class_qualifier_from_fd = row.get('fd_actual_class_qualifier')
                actual_method_name_from_fd = row.get('fd_actual_method_name')
                row['actual_java_filename'] = None
                for k_round_idx_init in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx_init+1}'] = "NOT_RUN"
                row['nondex_consistency'] = "NOT_EVALUATED"
                current_identifier = (project_name_in_csv, full_test_name_input, original_sha_from_csv, str(generated_patch_code)[:100], source_file_csv_type)
                if current_identifier in processed_identifiers:
                    logger.info(f"Skipping already processed entry: {project_name_in_csv} - {full_test_name_input[:50]}... - SHA {original_sha_from_csv[:7] if original_sha_from_csv else 'N/A'}")
                    continue
                if source_file_csv_type == "laky_fix_181_gpt35_labeled.csv":
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "SKIPPED"
                    row['nondex_consistency'] = "SKIPPED"; writer.writerow(row); outfile.flush(); continue
                if not original_sha_from_csv or pd.isna(original_sha_from_csv) or original_sha_from_csv == "":
                    logger.error(f"Empty or invalid original_sha ('{original_sha_from_csv}') for row {i+1} ({project_name_in_csv}). Skipping.")
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "INVALID_SHA"
                    row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                try:
                    imports_to_add_str = row.get('fd_import', "[]")
                    if not imports_to_add_str or not isinstance(imports_to_add_str, str) or imports_to_add_str.lower() == "none": imports_to_add = []
                    else: imports_to_add = eval(imports_to_add_str) if imports_to_add_str.startswith("[") else [imports_to_add_str]
                    pom_dependencies_str = row.get('fd_pom', "None")
                    if not pom_dependencies_str or not isinstance(pom_dependencies_str, str) or pom_dependencies_str.lower() == "none": pom_dependencies_str = None
                except Exception as e:
                    logger.error(f"Error evaluating fd_import/fd_pom for row {i+1} ('{imports_to_add_str}', '{pom_dependencies_str}'): {e}. Using defaults.", exc_info=True)
                    imports_to_add = []; pom_dependencies_str = None
                project_dir = os.path.join(FLAKY_REPOS_DIR, project_name_in_csv)
                logger.info(f"Checking out original SHA: {original_sha_from_csv} for project {project_name_in_csv} from {repo_url}")
                checkout_successful = git_checkout_commit(project_dir, original_sha_from_csv, repo_url, project_name_in_csv)
                if not checkout_successful:
                    logger.error(f"Failed to checkout SHA '{original_sha_from_csv}' for row {i+1}. Skipping subsequent operations for this row.")
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "GIT_CHECKOUT_FAIL"
                    row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                project_dir = os.path.join(FLAKY_REPOS_DIR, project_name_in_csv)
                git_clean_repo(project_dir)
                java_file_path, actual_java_fname = get_java_file_path_and_actual_name(
                    project_dir, module,
                    actual_class_qualifier_from_fd,
                    full_test_name_input,
                    source_file_csv_type
                )
                row['actual_java_filename'] = actual_java_fname
                if not java_file_path :
                    logger.error(f"Java test file not located (path is None) for {full_test_name_input} in {project_dir}/{module}. Row {i+1}. Skipping.")
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "FILE_NOT_FOUND_NONE_PATH"
                    row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                if not os.path.exists(java_file_path):
                    logger.error(f"Java test file path {java_file_path} does not exist for {full_test_name_input}. Row {i+1}. Skipping.")
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "FILE_NOT_FOUND_PATH_INVALID"
                    row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                logger.info(f"Located Java file: {java_file_path} (Actual name: {actual_java_fname})")
                original_java_content = ""
                try:
                    with open(java_file_path, 'r', encoding='utf-8', errors='replace') as f_orig:
                        original_java_content_read = f_orig.read()
                        if original_java_content_read is None :
                             logger.error(f"Reading original Java file {java_file_path} resulted in None. Skipping.")
                             for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "FILE_READ_NONE"; row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                        original_java_content = original_java_content_read
                except Exception as e:
                    logger.error(f"Could not read original Java file {java_file_path}: {e}. Skipping.");
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "FILE_READ_ERROR"; row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                method_to_patch = actual_method_name_from_fd if actual_method_name_from_fd else get_fqn_parts_from_input_csv(full_test_name_input, source_file_csv_type)[1]
                if not method_to_patch:
                    logger.error(f"Method to patch could not be determined for {full_test_name_input}. Skipping patch application.")
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "METHOD_NAME_PARSE_FAIL"; row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                if not apply_code_patch(java_file_path, original_java_content, generated_patch_code,
                                        method_to_patch,
                                        source_file_csv_type):
                    logger.error(f"Failed to apply code patch for {full_test_name_input} (method: {method_to_patch}). Restoring original file and skipping.");
                    if java_file_path and os.path.exists(os.path.dirname(java_file_path)):
                        try:
                            with open(java_file_path, 'w', encoding='utf-8', errors='replace') as f_restore: f_restore.write(original_java_content)
                        except Exception as e_restore_patch: logger.error(f"Error restoring file {java_file_path} after patch apply fail: {e_restore_patch}")
                    else: logger.error(f"Cannot restore file (patch apply fail), java_file_path is invalid: {java_file_path}")
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "PATCH_APPLY_FAIL"; row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                current_applied_imports = list(imports_to_add) if isinstance(imports_to_add, list) else []
                if not add_imports_to_java_file(java_file_path, current_applied_imports):
                    logger.warning(f"Initial import add for {full_test_name_input} failed/had issues.")
                pom_file_path = find_pom_xml(project_dir, module)
                original_pom_content = None; pom_modified = False
                if pom_dependencies_str and pom_dependencies_str.strip() and pom_dependencies_str.lower() != "none":
                    if pom_file_path and os.path.exists(pom_file_path):
                        try:
                            with open(pom_file_path, 'r', encoding='utf-8', errors='replace') as f_pom_orig: original_pom_content = f_pom_orig.read()
                            if safe_add_dependencies_to_pom(pom_file_path, pom_dependencies_str): pom_modified = True
                            else: logger.warning(f"safe_add_dependencies_to_pom returned false for {project_name_in_csv}.")
                        except Exception as e: logger.error(f"Could not read/write POM file {pom_file_path}: {e}", exc_info=True)
                    else: logger.warning(f"POM file {pom_file_path if pom_file_path else 'path not determined'} not found. Cannot add deps.")
                debug_file_name_before_build = f"DEBUG_BEFORE_BUILD_{project_name_in_csv.replace('/', '_')}_{full_test_name_input.replace('.', '_')}_{original_sha_from_csv[:7]}.java"
                debug_dest_file_before_build = os.path.join(DEBUG_JAVA_FILES_DIR, debug_file_name_before_build)
                try: shutil.copyfile(java_file_path, debug_dest_file_before_build)
                except Exception as e_debug_save: logger.error(f"Error saving pre-build debug Java file: {e_debug_save}")
                logger.info(f"Attempting initial build for {project_name_in_csv} module {module} after patching.")
                build_command = ["clean", "install", "-DskipTests",
                                 "-Dmaven.javadoc.skip=true", "-Drat.skip=true", "-Dcheckstyle.skip=true", "-Dfindbugs.skip=true",
                                 "-Denforcer.skip=true", "-Dspotbugs.skip=true", "-Djacoco.skip=true",
                                 "-Dspotless.check.skip=true", "-Dmaven.test.failure.ignore=true", "-DfailIfNoTests=false"]
                initial_build_ok, build_stdout, build_stderr = run_maven_command(project_dir, build_command, module)
                build_success_after_stitching = False
                if initial_build_ok:
                    build_success_after_stitching = True; logger.info(f"Initial build successful for {project_name_in_csv}.")
                else:
                    logger.warning(f"Initial build failed for {project_name_in_csv}. Attempting import stitching.")
                    content_before_stitching = ""
                    try:
                        with open(java_file_path, 'r', encoding='utf-8', errors='replace') as f_stitch_in: content_before_stitching = f_stitch_in.read()
                    except Exception as e_read_stitch:
                        logger.error(f"Could not read Java file for stitching {java_file_path}: {e_read_stitch}");
                        for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "STITCH_READ_ERROR"; row['nondex_consistency'] = "ERROR"; writer.writerow(row); outfile.flush(); continue
                    class_qualifier_for_stitch_param = actual_class_qualifier_from_fd if actual_class_qualifier_from_fd else get_fqn_parts_from_input_csv(full_test_name_input, source_file_csv_type)[0]
                    method_name_for_stitch_param = actual_method_name_from_fd if actual_method_name_from_fd else get_fqn_parts_from_input_csv(full_test_name_input, source_file_csv_type)[1]
                    nondex_test_param_for_stitch = ""
                    if class_qualifier_for_stitch_param and method_name_for_stitch_param:
                        true_class_qualifier_st = class_qualifier_for_stitch_param
                        if actual_java_fname and actual_java_fname.endswith(".java"):
                            simple_class_name_correct_case_st = actual_java_fname[:-5]
                            package_of_class_st = ".".join(true_class_qualifier_st.split('.')[:-1]) if '.' in true_class_qualifier_st else ""
                            if package_of_class_st : true_class_qualifier_st = f"{package_of_class_st}.{simple_class_name_correct_case_st}"
                            else : true_class_qualifier_st = simple_class_name_correct_case_st
                        nondex_test_param_for_stitch = f"{true_class_qualifier_st}#{method_name_for_stitch_param}"
                    if not nondex_test_param_for_stitch:
                        logger.error(f"Cannot perform stitching, NonDex param invalid for {full_test_name_input}"); build_success_after_stitching = False
                    else:
                        stitching_succeeded, _, stitched_imports = attempt_import_stitching(
                            java_file_path, project_dir, module, nondex_test_param_for_stitch, "8",
                            content_before_stitching, build_stdout, build_stderr
                        )
                        if stitching_succeeded:
                            build_success_after_stitching = True; logger.info(f"Import stitching successful for {java_file_path}.")
                            newly_stitched_imports = [imp for imp in stitched_imports if imp not in current_applied_imports]
                            if newly_stitched_imports:
                                current_applied_imports.extend(newly_stitched_imports); row['fd_import'] = str(current_applied_imports)
                                add_imports_to_java_file(java_file_path, current_applied_imports)
                            debug_file_name_stitched = f"DEBUG_STITCHED_{project_name_in_csv.replace('/', '_')}_{full_test_name_input.replace('.', '_')}_{original_sha_from_csv[:7]}.java"
                            try: shutil.copyfile(java_file_path, os.path.join(DEBUG_JAVA_FILES_DIR, debug_file_name_stitched))
                            except Exception as e_s_save: logger.error(f"Error saving stitched debug file: {e_s_save}")
                        else:
                            logger.error(f"Import stitching did NOT resolve build errors for {java_file_path}.")
                            debug_file_name_after_stitch_fail = f"DEBUG_STITCH_FAILED_{project_name_in_csv.replace('/', '_')}_{full_test_name_input.replace('.', '_')}_{original_sha_from_csv[:7]}.java"
                            try: shutil.copyfile(java_file_path, os.path.join(DEBUG_JAVA_FILES_DIR, debug_file_name_after_stitch_fail))
                            except Exception as e_sf_save: logger.error(f"Error saving post-stitch-fail debug file: {e_sf_save}")
                if not build_success_after_stitching:
                    logger.error(f"Failed to build {project_name_in_csv} finally. Skipping NonDex.")
                    final_build_status_msg = "STITCH_BUILD_FAIL" if not initial_build_ok else "BUILD_FAIL_NO_STITCH_ATTEMPT"
                    for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = final_build_status_msg
                    row['nondex_consistency'] = "ERROR"
                else:
                    try:
                        safe_project_name = "".join(c if c.isalnum() else "_" for c in project_name_in_csv)
                        safe_test_name = "".join(c if c.isalnum() else "_" for c in full_test_name_input)
                        patched_file_name_final = f"{safe_project_name}_{safe_test_name}_{original_sha_from_csv[:7]}.java"
                        dest_patched_file_final = os.path.join(PATCHED_JAVA_FILES_DIR, patched_file_name_final)
                        shutil.copyfile(java_file_path, dest_patched_file_final)
                        logger.info(f"Saved successfully built Java file to {dest_patched_file_final}")
                    except Exception as e_save_final: logger.error(f"Error saving final patched Java file: {e_save_final}")
                    nondex_round_outcomes = []
                    nondex_test_name_param = ""
                    method_name_for_nondex_run = actual_method_name_from_fd if actual_method_name_from_fd else get_fqn_parts_from_input_csv(full_test_name_input, source_file_csv_type)[1]
                    class_qualifier_for_nondex_run = actual_class_qualifier_from_fd if actual_class_qualifier_from_fd else get_fqn_parts_from_input_csv(full_test_name_input, source_file_csv_type)[0]
                    if method_name_for_nondex_run and class_qualifier_for_nondex_run and actual_java_fname:
                        simple_class_name_correct_case_nd = actual_java_fname[:-5]
                        package_of_class_nd = ".".join(class_qualifier_for_nondex_run.split('.')[:-1]) if '.' in class_qualifier_for_nondex_run else ""
                        if package_of_class_nd:
                            nondex_test_name_param = f"{package_of_class_nd}.{simple_class_name_correct_case_nd}#{method_name_for_nondex_run}"
                        else:
                            nondex_test_name_param = f"{simple_class_name_correct_case_nd}#{method_name_for_nondex_run}"
                        logger.info(f"Refined NonDex param using actual_java_fname: {nondex_test_name_param}")
                    else:
                        logger.warning(f"Could not form precise NonDex param for {full_test_name_input} (actual_fname='{actual_java_fname}', cq_fd='{actual_class_qualifier_from_fd}', mn_fd='{actual_method_name_from_fd}'). Using fallback.")
                        class_qualifier_nd_fb, method_name_nd_fb = get_fqn_parts_from_input_csv(full_test_name_input, source_file_csv_type)
                        if class_qualifier_nd_fb and method_name_nd_fb:
                             nondex_test_name_param = f"{class_qualifier_nd_fb}#{method_name_nd_fb}"
                             logger.info(f"Using fallback NonDex param: {nondex_test_name_param}")
                    if not nondex_test_name_param:
                        logger.error(f"Could not form NonDex test name for {full_test_name_input}. Skipping NonDex.")
                        for k_round_idx in range(NONDEX_INTERNAL_RUNS_COUNT): row[f'nondex_round{k_round_idx+1}'] = "NONDEX_NAME_ERROR"
                        row['nondex_consistency'] = "ERROR"
                    else:
                        overall_status, collected_details_list = run_nondex_test(
                            project_dir, module, nondex_test_name_param,
                            jdk_version="8", nondex_internal_runs_count_str=str(NONDEX_INTERNAL_RUNS_COUNT)
                        )
                        if len(collected_details_list) < NONDEX_INTERNAL_RUNS_COUNT:
                            logger.warning(f"NonDex returned {len(collected_details_list)} details, expected {NONDEX_INTERNAL_RUNS_COUNT}. Filling with status: {overall_status}")
                            fill_value = overall_status if overall_status not in ["COLLECTED_RUNS", "COLLECTED_RUNS_INCOMPLETE", "TEST_PASS_OVERALL", "TEST_FAILURE_OVERALL"] else "MISSING_DETAIL"
                            collected_details_list.extend([fill_value] * (NONDEX_INTERNAL_RUNS_COUNT - len(collected_details_list)))
                        elif len(collected_details_list) > NONDEX_INTERNAL_RUNS_COUNT:
                            logger.warning(f"NonDex returned {len(collected_details_list)} details, expected {NONDEX_INTERNAL_RUNS_COUNT}. Truncating.")
                            collected_details_list = collected_details_list[:NONDEX_INTERNAL_RUNS_COUNT]
                        for k_idx in range(NONDEX_INTERNAL_RUNS_COUNT):
                            detail_line = collected_details_list[k_idx] if k_idx < len(collected_details_list) else "MISSING_DETAIL_UNEXPECTED"
                            row[f'nondex_round{k_idx+1}'] = detail_line
                            current_outcome = "error"
                            if isinstance(detail_line, str):
                                match_detail = re.search(r"Failures:\s*(\d+),\s*Errors:\s*(\d+)", detail_line)
                                if match_detail:
                                    failures = int(match_detail.group(1)); errors = int(match_detail.group(2))
                                    current_outcome = "fail" if (failures > 0 or errors > 0) else "pass"
                                elif detail_line.startswith("NonDex summary: Passed"): current_outcome = "pass"
                                elif detail_line.startswith("NonDex summary: Flaky"): current_outcome = "fail"
                                elif detail_line in ["MAVEN_ARG_ERROR", "COMPILATION_ERROR", "BUILD_FAILURE",
                                                     "NO_TESTS_EXECUTED", "NONDEX_SCRIPT_MISSING",
                                                     "UNKNOWN_NONDEX_RESULT", "TIMEOUT_ERROR", "EXECUTION_ERROR",
                                                     "PARSE_ERROR", "MISSING_NONDEX_INTERNAL_RUN_SUMMARY",
                                                     "No Tests Executed By NonDex (summary)",
                                                     "NonDex BUILD FAILURE (no specific test line)",
                                                     "MISSING_DETAIL_UNEXPECTED",
                                                     "Build Failure (No Tests/Summary)"]:
                                    current_outcome = "error"
                            nondex_round_outcomes.append(current_outcome)
                            logger.info(f"NonDex internal run {k_idx+1} interpreted outcome: {current_outcome}, from detail: {detail_line if detail_line else 'N/A'}")
                        valid_outcomes = [o for o in nondex_round_outcomes if o in ["pass", "fail"]]
                        if not valid_outcomes: row['nondex_consistency'] = "ERROR_IN_ALL_NONDEX_RUNS"
                        elif len(set(valid_outcomes)) == 1: row['nondex_consistency'] = 1
                        else: row['nondex_consistency'] = 0
                        logger.info(f"NonDex consistency for {full_test_name_input}: {row['nondex_consistency']} (Outcomes for consistency: {valid_outcomes})")
                writer.writerow(row); outfile.flush()
                logger.info(f"Restoring original Java file: {java_file_path}")
                with open(java_file_path, 'w', encoding='utf-8', errors='replace') as f_restore: f_restore.write(original_java_content)
                if pom_modified and original_pom_content and pom_file_path and os.path.exists(pom_file_path):
                    logger.info(f"Restoring original POM file: {pom_file_path}")
                    with open(pom_file_path, 'w', encoding='utf-8', errors='replace') as f_pom_restore: f_pom_restore.write(original_pom_content)
                if (i + 1) % 20 == 0:
                    logger.info(f"Performing intermediate git clean for {project_name_in_csv}")
                    git_clean_repo(project_dir)
    logger.info("Stage 2: Patch application and NonDex validation finished.")
if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(DEBUG_JAVA_FILES_DIR).mkdir(parents=True, exist_ok=True)
    Path(PATCHED_JAVA_FILES_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("--- ENSURE YOU WANT TO RUN STAGE 1 PREPROCESSING ---")
    preprocess_data()
    logger.info("--- STARTING STAGE 2 PATCH APPLICATION AND VALIDATION ---")
    apply_and_validate()
    logger.info("All processing finished.")