// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

#ifdef  WIN32
#include <windows.h>
#include <stdlib.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfo
#include <stdio.h>
#include "processthreadsapi.h"

// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static void DebugMemoryInfo(const char* header)
{
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
    {
        //The following printout corresponds to the value of Resource Memory, respectively
        printf("%s:\tCommit \t\t\t=  0x%08X- %u (MB)\n", header, pmc.PrivateUsage, pmc.PrivateUsage / (1024*1024));
        printf("%s:\tWorkingSetSize\t\t\t=  0x%08X- %u (MB)\n", header, pmc.WorkingSetSize, pmc.WorkingSetSize / (1024 * 1024));
        printf("%s:\tPrivateWorkingSetSize\t\t\t=  0x%08X- %u (MB)\n", header, pmc.PrivateWorkingSetSize, pmc.PrivateWorkingSetSize / (1024 * 1024));
    }
}
# else
#include <iostream>
#include <fstream>
#include <string>
static void DebugMemoryInfo(const char* header) {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    long memory_usage = -1;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            // Extract the memory usage value (in kB)            
            sscanf(line.c_str(), "VmRSS: %ld kB", &memory_usage);
            break;
        }
    }
    if (memory_usage != -1) {
        std::cout << " Memory usage: " << header << ": " << memory_usage / (1024) << " MB" << std::endl;
    } else {
        std::cerr << " Failed to retrieve memory usage " << header << std::endl;
    }
}
#endif //  WIN32

int main(int argc, char* argv[]) try {
    if (4 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS_FILE> \"<PROMPT>\"");

    const std::string prompt("患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%、嗜酸性粒细胞（E）1%～5%、嗜碱性粒细胞（B）0～1%、淋巴细胞（L）20%～40%、单核细胞（M）3%～8%；临床意义：1）中性粒细胞为血液中的主要吞噬细胞，在细菌性感染中起重要作用。2）嗜酸性粒细胞①减少见于伤寒、副伤寒、大手术后、严重烧伤、长期用肾上腺皮质激素等。②增多见于过敏性疾病、皮肤病、寄生虫病，一些血液病及肿瘤，如慢性粒细胞性白血病、鼻咽癌、肺癌以及宫颈癌等；3）嗜碱性粒细胞 a 减少见于速发型过敏反应如过敏性休克，肾上腺皮质激素使用过量等。b 增多见于血液病如慢性粒细胞白血病，创伤及中毒，恶性肿瘤，过敏性疾病等；4）淋巴细胞 a 减少多见于传染病的急性期、放射病、细胞免疫缺陷病、长期应用肾上腺皮质激素后或放射线接触等。b 增多见于传染性淋巴细胞增多症、结核病、疟疾、慢性淋巴细胞白血病、百日咳、某些病毒感染等；5）单核细胞增多见于传染病或寄生虫病、结核病活动期、单核细胞白血病、疟疾等。5. 病人血小板计数结果：91 x10^9/L. 附血小板计数正常参考范围：（100～300）×10^9/L. 临床意义：1）血小板计数增高见于真性红细胞增多症、出血性血小板增多症、多发性骨髓瘤、慢性粒细胞性白血病及某些恶性肿瘤的早期等；2）血小板计数减低见于 a 骨髓造血功能受损，如再生障碍性贫血，急性白血病；b 血小板破坏过多，如脾功能亢进；c 血小板消耗过多，如弥散性血管内凝血等。6. 以往病例分析内容参考：白血病一般分为急性白血病和慢性白血病。1）急性白血病血常规报告表现为：白细胞增高，少数大于100×10^9/L，称为高白细胞白血病，部分患者白细胞正常或减少，低者可小于1.0×10^9/L，以AML中的M3型多见。在白细胞分类中，80％以上可见大量的幼稚细胞，有时仅见幼稚细胞和少量成熟的细胞，而无中间型细胞，称为白血病的裂孔现象。少数白细胞低的患者周围血幼稚细胞很少，此类患者必须骨髓穿刺才能确诊。多数急性白血病患者初诊时有不同程度的贫血；一般属正常细胞正色素型。但贫血很快会进行性加重。30％的患者血涂片中可见有核红细胞。血小板计数绝大部分患者减少，严重者小于10×10^9/L，仅极少数患者血小板计数正常。2） 慢性白血病血常规报告表现为：白细胞总数明显增高，通常大于30×10^9/L。半数患者大于100×10^9/L。中性粒细胞明显增多，可见各阶段粒细胞，以中性中幼粒，晚幼粒细胞居多，原始粒细胞小于等于10％，通常为1％～5％，嗜酸和嗜碱粒细胞亦增多。初诊时约有50％患者血小板增高，少数可大于1000×10^9/L。红细胞和血红蛋白一般在正常范围，若出现血红蛋白减低，血小板计数明显升高或降低，则提示疾病向加速期或急变期转化。7. 历史相关研究: 急性髓系白血病（AML）是造血干细胞恶性克隆性疾病。在AML的诊断、治疗以及判断预后的过程中，基因异常是一项重要指标。随着基因检测技术的不断进步，越来越多与AML发生相关的基因被人们发现，并且这些基因在指导预后方面有重要意义。常见和急性髓系白血病相关的基因突变有: 1）RUNX1-RUNX1T1：8号染色体和21号染色体易位[t(8;21)(q22;q22)]是RUNX1/RUNX1T1融合基因产生的基础。RUNX1/RUNX1T1融合基因与CBFB-MYH11融合基因导致的AML病因、临床特征相似，被合称为CBF-AML。核心结合因子（CBF）在造血干细胞的产生以及造血过程中起到重要作用，RUNX1编码CBF中的ɑ亚基，该亚基负责与DNA直接结合。因此RUNX1/RUNX1T1融合基因的产生会破坏CBF的功能，导致髓系分化阻断并最终导致白血病。2）CBFB-MYH11：CBFB-MYH11融合基因是染色体重排的结果，较常见inv(16)(p13.1q22)，较少见的类型为t(16;16)(p13.1;q22)。该融合基因与M4型AML发生相关，其特征为存在骨髓单核细胞母细胞和非典型嗜酸性粒细胞。小鼠模型表明，CBFB-MYH11融合基因可以破坏核心结合因子（CBF）的功能，导致髓系分化阻断并最终导致白血病。虽然单纯CBFB-MYH11的表达不足以导致白血病的发生，但CBFB-MYH11和其他突变的结合可以特异性地导致髓系白血病的发展。3）NPM1：核磷蛋白1（nucleophosmin 1，NPM1）属于核磷蛋白家族，是一种广泛表达的磷蛋白，能在核仁、核质和胞质之间不断穿梭。该基因位于5q35，包含12个外显子，编码3种核磷蛋白亚型。NPM1主要有4种功能：a 参与核糖体生物合成；b 维持基因的稳定性；c 依赖p53的应激反应；d 通过ARF-p53的相互作用从而调控生长抑制途径。4）CEBPA：CCAAT增强子结合蛋白α基因（CCAAT/en－hancer binding protein α，CEBPA）属亮氨拉链转录因子家族，位于染色体19q13。CEBPA突变会上调造血干细胞归巢和粒细胞分化的基因，下调参与调控造血细胞增殖的信号分子和转录因子的基因，阻碍DNA从G1期向S期演变，且诱导晚期造血细胞成熟，导致白血病发生。CEBPA基因突变发生率在成人AML患者中占5%-14%。突变可分为双突变和单突变，N端移码突变和C端框内突变同时存在即双突变较多见，而单杂合子突变不常见。国内外的多项研究表明双突变的预后良好，无论是CR率及维持化疗后的总CR率CEBPA双突变组均明显高于CEBPA单体突变组及CEBPA阴性组，且中位OS（60个月）和中位EFS（53个月）较其余两组均明显延长。5）MLLT3-KMT2A：MLLT3-KMT2A融合基因是t(9;11)(p21.3;q23.3)染色体易位形成的，其中赖氨酸甲基转移酶2A即KMT2A（旧称MLL）基因突变在AML中较常见，发生率约为10%。其中KMT2A与MLLT3（也被称为AF9或LTG9）基因融合是其中最常见的类型。因相较其他类型KMT2A融合基因的AML，MLLT3-KMT2A融合基因阳性的AML预后差异明显，因此将其单独列出。5）DEK-NUP214：DEK-NUP214融合基因由t(6;9)(p22.3；q34.1)引起，与大概1%的AML发生相关。Carl Sandén等人的研究发现DEK-NUP214基因主要影响细胞增殖过程，通过上调雷帕霉素复合物1(mTORC1)的活性来促进细胞增殖，使用雷帕霉素受体抑制剂治疗对抑制此类白血病细胞增殖有一定效果。DEK-NUP214融合基因阳性的AML通常预后不佳，Slovak所做队列研究显示该类型AML（包括儿童及成人）CR率仅有65%，中位OS仅有13.5个月，中位DFS仅有9.9个月。6）KMT2A：KMT2A基因（也称为MLL基因），位于11q23，编码组蛋白H3赖氨酸4甲基转移酶。KMT2A基因重排发生于大约3%-7%的成人初发AML。KMT2A编码一种组蛋白甲基转移酶，它在胚胎发育和造血过程中对基因表达的维持有重大作用。KMT2A基因易位会产生嵌合的KMT2A融合蛋白，直接与DNA结合并上调基因转录，导致下游KMT2A靶点的异常表达，包括HOX基因等，从而导致AML的发生。问：请基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、相关的基因突变、常用的治疗手段，以及当前已上市和正在临床阶段的药物清单。答：");

    std::string models_path = argv[1];
    std::string adapter_path = argv[2];
    int lora_id = atoi(argv[3]);
    std::string device = "GPU";  // GPU can be used as well
    int iter_num = 0;

    using namespace ov::genai;

    std::cout << ov::get_openvino_version() << std::endl;

    DebugMemoryInfo("Start with ");

    if (lora_id > 99) {
        LLMPipeline pipe(models_path, device, {ov::cache_dir("llm_cache")});  // register all required adapters here

        DebugMemoryInfo("Create base pipe ");

        std::cout << "Generate without lora :" << std::endl;
        while (iter_num < 10)
        {
            std::cout << pipe.generate(prompt, max_new_tokens(100)) << std::endl;
            DebugMemoryInfo("After inference base ");
            iter_num++;
        }

        return 0;
    }

    Adapter adapter(adapter_path);
    
    DebugMemoryInfo("Add adapter A ");

    LLMPipeline pipe(models_path, device, {adapters(adapter), ov::cache_dir("lora_llm_cache")});    // register all required adapters here

    DebugMemoryInfo("Create adapter pipe ");

    // Resetting config to set greedy behaviour ignoring generation config from model directory.
    // It helps to compare two generations with and without LoRA adapter.
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    pipe.set_generation_config(config);

    std::cout << "Generate with LoRA adapter and alpha set to 0.75:" << std::endl;
    while ( iter_num < 10 ) {
        std::cout << pipe.generate(prompt, max_new_tokens(100), adapters(adapter, 0.25)) << std::endl;

        DebugMemoryInfo("After inference Lora A ");
        iter_num++;
    }

    // pipe.remove_adapters(adapters(adapter));

    // DebugMemoryInfo("After remove Lora A ");

    // Adapter adapter1("C:\\model\\MiniCPM_lora\\COT_adapter_fp16\\minicpm-1b_lora\\adapter_model.safetensors");

    // DebugMemoryInfo("Load Lora B ");

    // std::cout << "Generate with LoRA adapter 111 and alpha set to 0.75:" << std::endl;
    // std::cout << pipe.generate(prompt, max_new_tokens(100), adapters(adapter1, 0.25)) << std::endl;
    
    // DebugMemoryInfo("After inference Lora B ");


    // pipe.remove_adapters(adapters(adapter1));

    // DebugMemoryInfo("After remove Lora B ");

    // std::cout << "\n-----------------------------";
    // std::cout << "\nGenerate without LoRA adapter:" << std::endl;
    // std::cout << pipe.generate(prompt, max_new_tokens(100), adapters()) << std::endl;
    
    // DebugMemoryInfo("After inference without Lora ");

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
