def Alltopics():
    topics = ['mediastinal germ cell tumor', 'expert testimony', 'hds', 'base of skull',
                'mpr', 'dracunculiasis', 'guinea worm', 'retinal haemorrhage', 'adenosine 5 triphosphatase', 'diane',
                'cystic acne', 'ois', 'agvhd', 'viiia', '5 dfurd', 'furd', 'mag3', 'food hypersensitivity', 'pctl', 
                'as2', 'kartagener s syndrome', 'summer season', 'taq', 'atypical neuroleptic', 'anterior cingulate',
                'acute respiratory distress syndrome', 'circularity', 'mutase', 'adrenergic blocking drug', 
                'systematic desensitization', 'the turning point', '9l', 'pyridazine', 'bisoprolol', 'trq',
                'propylhexedrine', 'type 18', 'darpp 32', 'rickettsia conorii', 'sport shoe', 'nervus terminalis',
                'somatostatinoma', 'ish', 'lagomorph', 'fimbriation', 'ng h', 'inotrope', 'pdp', 'stress relaxation',
                'metiamide', 'quisqualate receptor', 'grande', 'cex', 'antiparkinsonian drug', 'phase iv', 
                'p tyramine', 'beta casomorphin', 'icdh', 'anti bacterial', 'acsa', 'gabaa receptor antagonist', 
                'bull frog', 'non zero', 'mbd', 'continuous flow system', 'pervasive developmental disorder', 'nyi', 
                'plasma substitute', 'xt', 'cdn', 'mycoflora', 'st2', 'rmc', 'copp', 'optical axis', 'nafamostat', 
                'formamidase', 'adrenodoxin reductase', 'dnak', 'dnaj', 'alpha cyclodextrin', 'iid', 'streptomycete',
                'filarioidea', 'ventricular drainage', 'mitral incompetence', 'nao', 'hydrosalpinx', 
                'early pregnancy factor', 'epf', 'krukenberg tumor', 'odontogenic tumor', 'homing receptor', 
                'bone demineralization', 'automated image analysis', 'safranin', 'guinea bissau', 
                'continuous reinforcement', 'infidelity', 'etd', 'citrovorum factor', 'bcd', 'beta d galactose', 
                'penumbra', 'banking', 'processing deficit', 'napa', 'pectoral', 'human t cell lymphotropic virus type i', 
                'nuclear lamins', 'blood viscosity', 'time 3', 'comma', 'carprofen', 'neuromedin b', 'blap', 'miaa', 
                'hyperesthesia', 'medullary chromaffin cell', 'hpfh', 'melanesians', 'hantaan virus', 'cga', 'thioglycollate broth', 'ncc', 'heteroclitic', 'cerevisiae', 
                'fish oil', 'ugm', 'nak', 'camp release', 'psychometric testing', 'internal hernia', 'gmc', 'falciform ligament', 'dental morphology', 'cartoon', 
                'dermacentor', 'coagulation factor viii', 'heavy chain immunoglobulin', 'iema', 'complicated uti', 'marburg', 
                'primary hyperlipoproteinemia', 'phosphagen', 'acetyl l carnitine', 'autotrophic', 'dental medicine', 'ozonation',
                'mrf', 'aldosterone antagonist', 'methylmalonic acid', 'vasolidation', 'levobunolol', 'cricopharyngeus', 'gmf',
                'reflex action', 'premotor cortex', 'pa 2', 'burnout syndrome', 'whiplash', 'indican', 'sme', 'lurcher', 'mpi', 
                'rhodopseudomonas palustris', 'sex dimorphism', 'gcc', 'usha', 'apurinic site', 'bacteriophage phi x174', 'anesthetic drug',
                'infinitely', 'open access', 'desulfovibrio desulfuricans', 'pur', 'st 1', 'd ring', 'inertness', 'r ii', 
                '1 3 propanediol', 'k5', 'proline rich protein', 'rna ligase', 'preprotachykinin', 'body odor', 'cockatiel', 
                'clinically proven', 'cephalin', 'elt', 'oxygen isotope', 'pact', 'moniliformin', 'subcommissural organ', 
                'c factor', 'lithogenesis', 'addition reaction', 'smear layer', '3 hydroxy 3 methylglutaryl coa', 
                'high dose estrogen', 'outer nuclear layer', 'paramyosin', 'wide angle', 'auramine o', 'drag', 'colt', 
                'california mastitis test', 'horton s disease', 'wetland', 'moving belt', 'difunctional', 'gastroduodenal artery', 
                'inositol metabolism', 'venus', 'lattice structure', 'pancreatic tail', 'wide local excision', 'photoisomerization', 
                'wrk', 'histidyl trna synthetase', 'irritable bowel', 'blood compatibility', 'division of labor', 'reflected light', 
                'fef', 'ab1', 'flm', 'chaotropic', 'ufh', 'kcct', 'imino acid', 'codeinone', 'psh', 'congenital esotropia', 
                'transacetylase', 'tgr', 'antifreeze glycoprotein', 'cosmic radiation', 'redistribute', 'ntli', 'mcps', 
                'cytochrome b559', 'human gastrointestinal tract', 'glomerular mesangium', 'pancreatic ductal adenocarcinoma', 
                'aqua', 'opz', 'eicosapentaenoate', 'sodium pertechnetate', 'co dehydrogenase', 'prenylamine', 'm7', 'zein', 
                'zoladex', 'woodward s', 'uteroferrin', 'stochastic process', 'process model', 'pys', 'family practitioner committee', 
                'nadph diaphorase', 'multiple cloning site', 'mua', 'vin', 'epinine', 'methylergometrine', 'psychiatric rehabilitation', 
                'scf', 'malignant external otitis', 'podophyllin', 'dna glycosylase', 'rat line', 'thiol disulfide exchange', 
                'class 4', 'pride', 'cost management', 'philanthropy', 'symphysis', 'cape', 'underprivileged', 'mineralisation', 
                'infectious diarrhoea', 'gynaecomastia', 'galactic', 'pyrite', 'titan', 'west coast', 'microfluorimetry', 
                'e ferol', 'cobaltous', 'dorsoventrally', 'circadian clock', 'timm', 'hesitation', 'vermal', 'ring dove', 
                'house mouse', 'proline racemase', 'gc content', 'nicely', 'partial homology', 'relay cell', 'cytotoxic edema', 
                'gor', 'cyclobutane', 'glycolate oxidase', 'methyl glucoside', 'butyryl', 'cytochrome d', 'eticlopride', 'bchl', 
                'tgc', 'non depolarizing', 'rnase iii', 'alpha hydroxyacid', 'gal n', 'phosphatidylserine decarboxylase', 
                'guanylate', 'oxyntomodulin', 'phosphofructokinase 1', 'rmn', 'dho', 'thaumatin', 'pr protein', 'hadar', 'cav', 
                'sitz bath', 'social psychology', 'dinosaur', 'siberia', 'canary', 'popliteal aneurysm', 'plasmodium vinckei', 
                'phenylpyruvate', 'coproporphyrinogen oxidase', 'filaggrin', 'phenylalkylamine', 'r body', 'chondrocranium', 
                'cetacea', 'tongue muscle', 'clutch size', 'evolutionary process', 'peromyscus', 'anolis', 'bluegill', 'slw', 
                'mycorrhizal', 'peromyscus maniculatus', 'reformed', 'thoracic ganglion', 'polycythaemia vera', 'cauliflower', 
                'ino', 'double layered', 'agria', 'cytochrome b6', 'ssu', 'chronic back pain', 'thoracic kyphosis', 'flatness', 
                'humic', 'harpoon', 'bagasse', 'aldose', 'matching law', 'ulp', 'light reaction', 'nitrogen assimilation', 
                'extensin', 'elicitors', 'linearis', 'sanguinarine', 'fucoxanthin', 'r band', 'zero point', 'guess', 'acorn', 
                'biofilms', 'inoculant', 'm sativa', 'lactic acid bacteria', 'l acidophilus', 'spikelet', 'decabromodiphenyl oxide', 
                'blue 1', 'jpn', 'protective service', 'oig', 'weinberg', 'g factor', 'positronium', 'dense connective tissue', 
                'cotton wool spot', 'cyclothymia', 'occupational stress', 'outcome data', 'supply demand', 'ubf', 'p r interval', 
                'ptn', 'tep', 'king county', 'north central', 'anglos', 'lateral sinus', 'glass model', 'shy drager syndrome', 
                'coturnix quail', 'sideroblast', 'uvc', 'mhp', 'bhp', 'detrusor hyperreflexia', 'non human', 'daba', 'hpp', 'tpo', 
                'upper motor neuron', 'm d anderson hospital', 'retinoschisis', 'ucl', 'cervical spinal injury', 'ckbb', 
                'aquatic organism', 'polyclinics', 'pregnancy prevention', 'hypermetropia', 'latent syphilis', 'iotrolan', 
                'galactosaemia', 'vitamin b6 deficiency', 'childless', 'lumbar disc herniation', 'american lobster', 
                'hepatopancreas', 'etu', 'schneiderian', 'pubic bone', 'vinpocetine', 'type 0', 'statistical inference', 
                'rebound effect', 'hayflick', 'rht', 'temocillin', 'strauss', 'sodium sulphate', 'pulsus paradoxus', 'm xenopi', 
                'sex therapy', 'lesbianism', 'hyperkalemic periodic paralysis', 'patellectomy', 'tth', 'artificial lung', 'cev', 
                'segmenting', 'ion chamber', 'flavour', 'spreadsheet', 'calculator', 'micro computer', 'magnetic tape', 
                'volume velocity', 'oxyntic cell', 'assaulted', 'w line', 'podocyte', 'sexual history', 'monoacylglycerol lipase', 
                'mycobacillin', 'rq', 'contrast effect', 'mycoplasmosis', 'western australian', 'p lp', 'acf', 'breast enlargement', 
                'antimicrobial chemotherapy', 'diffusion layer', 'phobic disorder', 'moslem', 'fragile x chromosome', 'the p group', 
                'lateral femoral condyle', 'knee arthroscopy', 'avoidance reaction', 'homeothermic', 'pulmonary veno occlusive disease', 
                'rectangle', 'zielke', 'cervical spondylotic myelopathy', 'dishabituation', 'alcoholic delirium', 'early puberty', 
                'paraprofessional', 'vulpes vulpes', 'udr', 'clot retraction', 'fis', 'fundic mucosa', 'inhalational anesthetic', 
                'crystal lattice', 'vulvovaginal', 'ma s', 'unit type', 'adrenal cortical carcinoma', 'club foot', 'nocardia brasiliensis', 
                'linamarin', 'sugar alcohol', 'liver steatosis', 'sex chromatin', 'tibolone', 'lynestrenol', 'apert syndrome', 'pmt', 
                'cscc', 'pxe', 'blood system', 'fibroplasia', 'puppet', 'mui', 'micro scale', 'wbr', 'strumal carcinoid', 'reanimation', 
                'mutacin', 'chronic pneumonia', 'upland', 'n dama', 'lvedd', 'chemical safety', 'non parametric', 'tbf', 'recidivism', 
                'fwhm', 'codeine phosphate', 'ukraine', 'egl', 'dzp', 'voice rehabilitation', 'ftb', 'lasix', 'reverse tolerance', 
                'sound intensity', 'electric fish', 'album', 'h pi', 'coccidian', 'insectivore', 'idh', 'kiv', 'mjd', 'explication', 
                'partridge', 'attention to detail', 'discitis', 'aminothiazole', 'masked depression', 'bc1', 'southern europe', 
                'aortic root dilatation', 'electron imaging', 'mule', 'brain disorder', 'cognitive appraisal', 'single line', 
                'gestodene', 'actinomadura', 'brcn', 'microaerophilic', 'kii', 'dd peptidase', 'organismal', 'bacterial spore', 
                'pall', 'hydrogen production', 'fmh', 'okan', 'growth promoter', 'cerebral atherosclerosis', 'phosphatidyl inositol', 
                'aoa', 'casein kinase', 'trichinellosis', 'tracheobronchial lymph node', 'citrinin', 'disincentive', 'articular process', 
                'post graduate', 'free association', 'diclofensine', 'drn', 'prenatal stress', 'zellweger s syndrome', 'brunetti', 
                'dai', 'haptics', 'ptns', 'csfp', 'lathosterol', 'schulz', 'lucite', 'chloropropanediol', 'subadult', 'cmo', 
                'glass wool', 'cyclohexene', '2 methylnaphthalene', 'yates', 'billroth', 'bcs', 'gcd', 'neonatal herpes', 
                'asthenozoospermia', 'ooplasm', 'slow cooling', 'rubeosis iridis', 'lc2', 'formula milk', 'malachite green', 
                'monier williams', 'thermal energy', 'ch2cl2', 'indeno', 'kpf', 'csw', 'sunset', 'vigabatrin', 'lacrimal duct', 
                'pathogenic microbe', 'kasabach merritt syndrome', 'prodrome', 'central serous retinopathy', 'hyperacuity', 
                'submental', 'eph', 'lgb', 'galanthamine', 'hph', 'brno', 'lrcs', 'echinomycin', 'congenital myopathy', 'anomic', 
                'homophone', 'fah', 'alley', 'proactive interference', 'convoluted tubule', 't 2 mycotoxin', 'x zone', 'gpx1', '5g', 
                'dysplastic nevus', 'pdhc', 'circulating endothelial cell', 'stroke unit', 'distal axonopathy', 'drp', 'srb', 
                'glomerular hyperfiltration', 'criterium', 'xj', 'consultation liaison psychiatry', 'lateral release', 
                'hemophilic arthropathy', 'spr', 'neural foramen', 'cmap', 'sugiura procedure', 'stomatological', 'uhf', 
                'attributional style', 'introvert', 'parietal peritoneum', 'waterston', 'pgn', 'tuberous', 'silenced', 'phytic acid', 
                'ulnar deviation', 'microphonics', 'acoustic emission', 'choriocarcinomas', 'stiff man syndrome', 'millon clinical multiaxial inventory', 
                'patellar dislocation', 'subluxated', 'chromobacterium violaceum', 's haemolyticus', 'psf', 'conventional wisdom', 
                'vienna austria', 'azq', 'tendon cell', 'air interface', 'trichuriasis', 'direct coombs test', 'hodgkin disease', 
                'yaya', 'yayc', 'copper binding protein', 'elevated alkaline phosphatase', '1 2 dce', 'dsf', 'pncb', 
                'drug eruption', 'jadassohn', 'skeletal disease', 'calcium 45', 'prednimustine', 'fibrocystic breast disease', 
                'duplication cyst', 'hammersmith', 'car safety', 'ampo', 'dbe', 'volcano', 'space motion sickness', 
                'university of iowa', 'portability', 'elastic bandage', 'transannular', 'atrial appendage', 'aortic valvuloplasty', 
                'dyskeratosis congenita', 'ky', 'manatee', 'svo2', 'piia', 'forced oscillation', 'metaproterenol sulfate', 'smds', 
                'incineration', 'bloodmeal', 'circannual rhythm', 'tertiary care center', 'cardiorespiratory fitness', 'ectrodactyly', 
                'cyclic neutropenia', 'osf', 'crista terminalis', 'xsa', 'llv', 'boari', 'gem', 'vel', 'transfusion reaction', 'nvg', 
                'isophorone', 'tilidine', 'nbg', 'nmg', 'ala nasi', 'solitary tract', 'hydroxy acid', 'ots', 'wernicke s area', 
                'brainstem glioma', 'thymine dimer', 'm component', 'class switching', 'taussig', 'bing', 
                'aortic valve insufficiency', 'lipoid', 'vascular constriction', 'hlt', 'csf pleocytosis', 'hickman', 
                'stapedius reflex', 'fungiform papilla', 'sciatic artery', 'clc', 'crane fly', 'ergoloid mesylates', 
                'halobacteria', 'methanobacterium formicicum', 'adolescent health', 'rod cell', 'ft2', 'dnt', 'extraskeletal myxoid chondrosarcoma', 
                'lmp', 'repartition', 'cspg', 'metroplasty', 'laca', 'tgb', 'nalmefene', 'flestolol', 'diketopiperazine', 'critical illness', 
                'linxian', 'monoxygenase', 'liposomal doxorubicin', 'benzo c phenanthrene', 'npyr', 'tbz', 'endogeneity', 
                'milk substitute', 'school curriculum', 'estrogen metabolism', 'social force', 'shoreline', 'the nursing record', 
                'infant weight', 's hominis', 'nti', 'rhabdoid tumor', 'aafp', 'cubic millimeter', 'ssa ro', 'cdc7', 'genius', 
                'duchenne s muscular dystrophy', 'ufc', 'celiac trunk', 'ipcs', 'fhf', 'magnification factor', 'new frontier', 
                'emetogenic', 'eck', 'sc3', 'physical form', 'tolnaftate', 'excess fluid', 'vitamin b1', 'smz', 'bismuth subsalicylate', 
                'overdenture', 'self directed learning', 'erms', 'coq10', 'non specific symptom', 'nurse researcher', 'reading speed', 
                'developmental dyslexia', 'o methyltransferase', 'ajoene', 'tumour antigen', 'perivitelline space', 'dupuytren s disease', 'myofibroblast', 
                'chymosin', 'opp', 'gpp', 'nalm', 'rmcp', 'xerosis', 'propidium', 'dfa', 'above knee amputation', 'sft', 'l lactate dehydrogenase', 
                'inhaled bronchodilator', 'hp1', 'folacin', 'dezocine', 'gamma trace', 'cystatin c', 'l myc', 'caulobacter crescentus', 
                'plant pathogen', 'gabhs', 'hexitol', 'dbd', 'elimination diet', 'light peak', 'hipa', 'o side chain', 'sclerotome', 
                'reactive lymphocyte', 'propagation time', 'subserosal', 'frf', 'oros', 'transverse sinus', 'cfu l', 'angle of incidence', 
                'pte', 'subsidized', 'mees', 'hcrh', 'reconstruction algorithm', 'provitamin', 'iip', '16 gauge', 'willebrand', 
                'chu', 'quitting', 'fluorine 18', 'rce', 'pba', 'rnu', 'brain trauma', 'pantopaque', 'doh', 'impaired growth', 
                'cold spot', 'bloom syndrome', 'fibrous dysplasia of bone', 'halitosis', 'cefpiramide', 'algesic', 
                'masticatory force', 'blood group system', 'butalbital', 'pain disorder', 'moxestrol', 'rbg', 'executing', 
                'otalgia', 'aspirin intolerance', 'medial pterygoid muscle', 'typhlitis', 'bowel rest', 'disseminated intravascular coagulopathy', 
                'lactose absorption', 'effective population size', 'urushiol', 'fracture toughness', 'metal oxide', 'tranilast', 
                'von recklinghausen neurofibromatosis', 'rickettsia rickettsii', 'tbpa', 'exacerbates', 'adult t cell lymphoma', 
                'point of interest', 'd8', 'blalock', 'thyroxine binding prealbumin', 'f xii', 'stumpy', 'culicoides', 'smd', 'mmpl', 
                'rhamnolipid', 'cefpimizole', 'kalam', 'coll', 'saguinus', 'fludrocortisone', 'anthrax toxin', 'ith', 'hunter syndrome', 
                'fibrous protein', 'bnc', 'combined oral contraceptive', 'rhinorrhoea', 'carbonated', 'thiomalate', 'nom', 'profibrinolytic', '5x', 
                'bow', 'galactosialidosis', 'glucuronosyltransferase', 'dka', 'piz', 'vj', 'amf', 'wfa', 'drug design', 'picu', 'thg', 
                'gravitropism', 'obstetric ultrasound', 'adherens junction', 'b anthracis', 'area 3', 'spermidine spermine n1 acetyltransferase', 
                'abg', 'tp5', 'becton dickinson', 'cfv', 'rch', 'red cell aplasia', 'sphi', 'trilostane', 'tc1', 'roentgen ray', 
                'synteny', 'iso 1', 'ltb', 'dunce', 'nmp', 'ferri', 'rela', 'fiac', 'bcv', 'tracheitis', 'opsoclonus', 'fnp', 
                'beta glycerophosphate', 'zwitterion', 'tall fescue', 'type 2 pneumocytes', 'cleft hand', 'clefting', 'ubiquitination', 
                'adpase', 'hirano body', 'halocarbon', 'pyp', 'lucerne', 'tcdf', 'dorsal interosseus', 'phosphatidylserine synthase', 
                'waymouth', 'sertoli cell tumor', 'ovine progressive pneumonia', 'man 6 p', 'hot flash', 'neutrophil collagenase', 'eee virus', 
                'eee', 'adrenal carcinoma', 'synaptophysin', 'va rna', 'mtv 2', 'mental foramen', 'act d', 'jun n virus', 'virus latency', 
                'bovine herpesvirus', 'hemp', 'penicillic acid', 'bradyrhizobium', 'psi value', 'am1', 'methanol dehydrogenase', 'ntr', 
                'puc18', 'aab', 'ead', 'aortic body', 'dr6', 'mineralocorticoid receptor', 'protein p53', 'endovascular treatment', 
                'childhood diabetes', 'closed angle glaucoma', 'gtf', 'smf', 'ufm', 'p day', 'phosphoglucomutase 1', 'lipotropin', 
                'b endorphin', 'b ep', 'pickup', 'backward masking', 'svl', 'fbf', 'adenosine deaminase inhibitor', 'etodolac', 
                'chitin synthetase', 'nafarelin', 'oligomenorrhea', 'bf1', 'expectorated', 'gran', 'gif', 'mammalian lh rh', 
                'h2 receptor blocker', 'bucco', 'fel', 'rural district', 'bank vole', 'corwin', 'buf', 'dtg', 'medullary pressor', 
                'gml', 'oxypertine', 'rs2', 'andreasen', '5ht1b', 'doxylamine', 'cfu m', 'cmb', 'rlt', 'bpo', 'nonbenzodiazepine', 
                'knee flexor', 'cellular replication', 'pirbuterol', 'trimeprazine', 'ipecac syrup', 'total mastectomy', 
                'phosphonium', 'antigen psa', 'sp5', 'blt', 'choroidal neovascularization', 'tmvp', 'hfp', 'hkn', 'spinous cell', 
                'lignoceric acid', 'electroreceptive', 'griseum centrale', 'thionin', 'senning', 'ra n', 'maxillary nerve', 
                'ganglionic blocker', 'alpha 1 adrenoreceptor', 'pi t', 'ivig', 'thalamic syndrome', 'delta k', 'paleostriatum', 
                'loc', 'bkd', 'lns', 'pmcs', 'multifidus', 'plasmodium cynomolgi', 'vrnf', 'taf', 'vp6', 'protein methylation', 
                'argininosuccinic aciduria', 'sm5', 'jbp', 'ehna', 'midodrine', 'mtg', 'sodium cyclamate', 'schedule 2', 
                'l deprenyl', 'c bovis', 'hub', 'c sec', 'oxytocinase', 'kawasaki s disease', 'tfl', 'recto', 'gp20', 
                'root fracture', 'mbbr', 'ldc', 'heparin induced thrombocytopenia', 'alpha ketoisovalerate', 'cerebral anoxia', 
                'skier', 'rheumatic disorder', 'lofentanil', 'p2 receptor', 'eugenics', 'convoluta', 'simultaneous release', 
                'naja atra', 'polyacrylate', 'bucindolol', 'lda', 'k8', 'ciclosporin', 'glycophorins', 'hmr', 'endophyte', 
                'ferriheme', 'phosphomonoesterase', 'floxuridine', 'radiocarbon', 'needlestick injury', 'meissner corpuscle', 
                'metal cluster', 'c h bond', 'aminobutyric acid', 'hrg', 'claustral', 'cancer family syndrome', 'tiropramide', 
                'ang 1', 'zi', 'international cometary explorer', 'comet giacobini zinner', 'tuberculostatic', 'bif', 'jacalin', 
                'pronator', 'triprolidine', 'discrete time', 'n acylurea', 'munsell', 'ibm personal computer', 'plap', 'adcp', 
                'whirlpool', 'bunyamwera virus', 's segment', 'interferon alfa 2b', 'heparan sulphate proteoglycan', 'allantoicase', 
                'antennal lobe', 'fractional synthesis', 'docking', 'g enzyme', 'lfa 3', 'deoxynucleotides', 'tso', 'hil', 'ffl', 
                'phosphoribulokinase', '2pi', 'aldolase c', 'irritable bladder', 'paracoccus', 'homoeopathic', 'chlorbutol', 
                'x ray microscopy', 'interstellar', 'parasponia', 'naked dna', 'linear acceleration', 'esperase', 'hpma', 
                'polycystic disease', 'uvrc', 'social health', 'the social network', 'meaning of life', 'bricker', 
                'malignant angioendotheliomatosis', 'ln2', 'pulp capping', 'stress fibre', 'dicyclomine', 'aadc', 
                's adenosylhomocysteine hydrolase', 'medetomidine', 'poly a polymerase', 'haemopexin', 'scv', 
                'sulfotransferases', 'ice cold', 'feline leukaemia virus', 'disassembled', 'er alpha', 'ae2', 
                'mucopurulent cervicitis', 'sulphanilamide', 's iv', 'mudminnows', 'oxygen 18', 'octadecadienoic acid', 
                'veto', 'nuv', 'angelicin', 'xanthotoxin', 'celery', 'molecular target', 'taipei', 'green plant', 'wfm',       
                'medicago', 'paniculata', 'dhm', 'mara', 'hybrid plant', 'varietal', 'carlsberg', 'transition zone', 'spurr', 
                'natural science', 'carbon hydrogen bond', 'hypertrophic scar', 'incisure', 'boll', '9 fluorenone', 'hydrothermal',
                'selective adsorption', 'skunk', 'galactitol', 'preconditioning', 'tx1', 'alternative way', 'carelessness', 'varimax rotation', 
                'methanethiol', 'tuberculosis control', 'n butyl chloride', 'hartmann s solution', 'sd 11', 'ocb', 'kaster', 'pedaling', 
                'ascorbic acid deficiency', 'pulse per second', 'coprecipitated', 'abz', 'lymph gland', 'african elephant', 'sfp', 'pdg', 'sufficiency', 
                'hq', 'vastus medialis', 'marijuana smoking', 'totp', 'ala dehydratase', 'local government', 'radiation carcinogenesis', 'mistreatment', 'dle', 'hypertensive encephalopathy', 'pnps', 'ketorolac', 'rubbed', 'pmd', 'highlander', 'palyam', 'dakar', 'linear regression model', 'nevoid basal cell carcinoma syndrome', 'cetp', 'zp3', 'pathological gambling', 'levan', 'alginates', 'relative motion', 'meibomian', 'clear cell sarcoma', 'saponifiable lipid', 'low calorie diet', 'inorganic arsenic', 'mast cell stabilizer', 'persistent light reaction', 'dlh', 'knossos', 'echinocytes', 'amsler grid', 'self rated health', 'tenant', 'cjc', 'spanish surname', 'sodium cyanate', 'prat', 'lorcainide', 'intracranial infection', 'premenarchal', 'slough', 'aah', 'internal auditory meatus', 'hc 3', 'bacteriocins', 'gibbon', 'udc', 't 47d', 'propyl gallate', 'lateral semicircular canal', 'statoconia', 'gr9', 'tla', 'yoga', 'testicular artery', 'ventricular pre excitation', 'march april', 'marasmic', 'recovering alcoholic', 'vge', 'ifb', 'major urinary protein', 'centrum semiovale', 
                'o zone', 'enamelins', 'amelogenins', 'layman', 'cvb', 'enteral formula', 'defaunation', 'caii', 'nucleoside phosphotransferase', 'lsc', 'abd', 'ppn', 'dimethoate', 'appalachian', 'htps', 'pib', '1 3 diaminopropane', 'i3c', 'phb', 'root level', 'pregnane', 'skin chamber', 'milker', '3 azido 3 deoxythymidine', 'bouquet', 'nga', 'diphenol', 'corn cob', 'sg 3', 'loss of hearing', 'memory formation', 'genetic method', 'glycerol kinase deficiency', 'methoxyl', 'phsa', 'aves', 'animal behaviour', 'snt', '25 hydroxylase', 'orthopoxviruses', 'ga2', 'hg l', 'trl', 'vitamin d resistant rickets', 'vp 10', 'centrophenoxine', 'matrix analysis', 'hvp', 'orip', 'agnoprotein', 'agnogene', 'bsf1', 'asw', 'rotavirus gastroenteritis', 'region iii', 'sfcm', 'pro7', 'pawp', 'fusarin c', 'blob', 'signal strength', 'linear velocity', 'amylose', 'script', 'biliary stent', 'frq', 'n methylspiperone', 'temporal horn', 'remoxipride', 'ameloblastic fibroma', 'tgg', 'vertical jump', 'gba', 'dextranase', 'mala', 'prtc', 'ectopic gestation', 'dydrogesterone', 'crassipes', 'ngfr', 'indenolol', 'png', '3 nucleotidase', 'lod', 'catha edulis', 'anisomycin', 'd aspartic acid', 'diamox', 'sgb', 'ggtp', 'transpeptidase', 'gp 25', 'o7', 'stylonychia', 'rhl', 'keratohyalin', 'south east asian', 'parietaria judaica', 'nwr', 'red tide', 'dmc', 'inosine pranobex', 'white ramus', 'ra3', 'podocalyxin', 'ciguatoxin', '3 ap', 'filiform papilla', 'key event', 'gye', 'unbalanced growth', 
                'dutpase', 'fim', 'prebound', 'efferent system', 'w5', 'pais', 'cgt', 'advanced prostate cancer', 'extrapyramidal system', 'odonto', 'abm', 'phenylpropionate', 'salt gland', 'sgf', 'biotinidase', 'trans splicing', 'mc3t3', 'bevantolol', 'tamp', 'pyruvate decarboxylase', 'alpha neurotoxin', 'lor', 'felis', 'isocortex', 'pheresis', 'nobel prize', 'mofebutazone', 'phycomyces', 'phycomyces blakesleeanus', 'coi', 'dumb', 'aprindine', 'gyrus dentatus', 'sf1', 'conn s syndrome', 'avi', 'vega', 'fct', 'isocyanide', 'beta hairpin', 'mitotane', 'lcf', 'iron sulfur', 'cr 7', 'cytochalasin e', 'pwc', 'ocular cicatricial pemphigoid', 'vba', 'udp gt', 'eet', 'iio', 'htnf', 'dominant hand', 'tnf beta', 'nh4oh', 'kna', 'aau', 'parasympatholytic', 's and l', 'denv', 'dar', 'mtbe', 'tile', 'psychological development', 'postcentral gyrus', 'marfan s disease', 'polio vaccine', 'cns disorder', 'usn', 'dppa', 'dlm', 'plateletpheresis', 'loxp', 'wh', 'idoa', 'nitroethane', 'vitamin k 3', 'arylsulfatase c', 'q6', 'mannosyltransferase', 'ihp', 'aj', 'uh', 'cosolvent', 'edr', 'smt', 'hrf', 'amia', 'sect', 'ramets', 'cucurbit', 'yucca moth', 'prothoracic gland', 'legumin', 'lsu', 'pj', 'mass extinction', 'opt', 'bottled water', 'sulphur compound', 'rfm', 'icap', 'trivalents', 'sg 1', 'food and nutrition', 'ceas', 'parr', 'metapramine', 'drill hole', 'mentality', 'oct 2', 'rti', 'abbr', 'nib', 'nbas', 'detriment', 'host computer', 'kanji', 'ferroxidase', 'ged', 'exophthalmometer', 'negative geotaxis', 'orthovoltage',
                 'alpha d mannosidase', 'togo', 'mean width', 'pulmonary artery sling', 'esophageal rupture', 'lva', 'bull calf', 'hpm', 'omv', 'eda', 'granite', 'balint', 'mycotoxin t 2', 'betel quid', 'aschoff', 'pika', 'intracranial haemorrhage', 'ept', 'ahf', 'exchangeability', 'kal', 'oxitropium bromide', 'femoxetine', 'exfoliation syndrome', 'tolfenamic acid', 'flier', 'delc', 'vacutainer', 'fast green', 'sodium lauryl sulphate', 'obsidan', 'chemie', 'aspergillus parasiticus', 'atii', 'nanocapsules', 'vldl receptor', 'oligophrenia', 'retinal hole', 'uer', 'nidd', 'dorso ventral', 'electrocution', 'neural plasticity', 'decamers', 'sexual identity', 'cil', 'nucleus ventralis', 'haa', 'todd', 'nutria', 'esf', 'damme', 'english speaking', 'embolizing', 'leg length inequality', 'pavlov', 'conditional reflex', 'right auricle', 'voxel', 'fem', 'septum primum', 'holocaust survivor', 'perceived control', 'dpti', 'sulphadiazine', 'streptomyces hygroscopicus', 'lichen ruber planus', 'intellectual function', 'pyrocatechol', 'fumarylacetoacetase', 'low carbohydrate diet', 'avermectin', 'paecilomyces', 'avermectins', 'magnetic resonance imaging of the brain', 'nephrogenic adenoma', 'parent training', 'retinoscopy', 'quinidine gluconate', 'fsd', 'mycobacterium gordonae', 'fragile x mental retardation', 'sick role', 'yogurt', 'simultaneity', 'long jump', 'alpha o', 'wj', 'postcoital test', 'allylestrenol', 'day care centre', 'kurtosis', 'chek', 'yield stress', 'cavia porcellus', 'wrongful life', 'torpid', 'mav', 'brushite', 'spruce budworm', 'docent', 'stepfamilies', 'vitek', 'customized', 'shf',
                'otoconia', 'microcins', 'rcf', 'german language', 'anencephalus', 'vaginal fluid', 'higher nervous activity', 
                't pll', 'dependant', 'biological clock', 'hierarchic', 'apm', 'og', 'indicator bacteria', 'pierced', 'icx', 'nq', 'rma', 'terodiline', 'kx antigen', 'isd', 'rrs', 'neuroendocrine differentiation', 'systemic sarcoidosis', 'oxytocin challenge test', 'stereotaxis', 'toadfish', 'role of government', 'phz', 'fetal hydrops', 'gingival margin', 'robotic', 'bifonazole', 'leader peptidase', 'hist', 'iga1 protease', 'erythrogenic toxin', 'angi', 'meade', 'coralline', 'multicystic dysplastic kidney', 'pa6', 'neuroreceptors', 'indalpine', 'erythromycin ethylsuccinate', 'fsc', 'cet', 'nmj', 'holtzman', 'prevertebral', 'melkersson rosenthal syndrome', 'edgewise', 'chancre', 'extrinsic allergic alveolitis', 'zinc oxide eugenol', 'fatty alcohol', 'alpha 1b', 'distance matrix', 'mli', 'pri', 'sccs', 'paca', 'thi', 'botany', 'cnu', 'helicoidal', 'panoxyl', 'xian', 'fo2', 'leukocidin', 'dermatan', 'lng', 'wca', 'serotonin n acetyltransferase', 'shs', 'htl', 'reproterol', 'phaeomelanin', 'nge', 'globoid', 'mossy fibre', 'vcs', 'fav', 'female to male transsexual', 'phytase', 'extrusion cooking', 'rvi', 'bdz', 'clz', 'l cycloserine', 'helf', 'sultamicillin', 'diazinon', 'emory', 'cd4 t cell', 'x linked recessive disorder', 'udpg', 'embryologic development', 'dil', 'primary hemostasis', 'knrk', 'gyra', 'type iv collagenase', 'occupational dermatitis', 'elr', 'ryazan province', 'whitlockite', 'efm', 'bag2', 'myotoxin', 'mctp', 'nmsp', 'alcaligenes eutrophus', 'pyocyanin', 'cornification', 'cowshed', 'tsig', 'hypothalamic dysfunction', 'bab', 'perazine', 'eos', 'acsf', 'hpk', 'axiom', 'plase', 'tft', 'tfc', 'ilm', 'fludarabine phosphate', 'pp cell', 'dmnt', 'thomsen friedenreich antigen', 'rpmc', 'opidn', 'cresyl', 'asm', 'black south african', 'schaffer', 'endangered specie', 's ren', 'allantoinase', 'ump synthase', 'eudistomin', 'ansaid', 'enkephalinase b', 'aluminosilicates', 'nailed', 'lipidex', 'fatherhood', 'mmp', 'ewl', 'rheumatoid vasculitis', 'hemianopsia', 'kb5', 'cape peninsula', 'lbd', 'bendroflumethiazide', 'sulfadoxine pyrimethamine', 'decimeter', 'ipc', 'signal pathway', 'jail', 'timidity', 'legitimacy', 'self organizing', 'bold', 'lower class', 'ando', 'cntf', 'smb', 'copeptin', 'myology', 'canid', 'red fox', 'beaver', 'grey hair', 'preening', 'elm', 'hmc', 'near death experience', 'dialectic', 'psychoanalytical', 'phosphate deficiency', 'xanthone', 'isoamyl alcohol', 'zeste', 'periclinal', 'sporangiophore', 'klebs', 'cutinase', 'o phenylphenol', 'gold toxicity', 'phentermine', 'brahmin', 'andhra pradesh', 'tectorial membrane', 'office automation', 'response variable', 'myofascial trigger point', 'interpersonal interaction', 'aminocaproic acid', 'munchausen syndrome by proxy', 'lhb', 'glottal', 'epiphysiodesis', 'dwyer', 'right main bronchus', 'u na', 'f1b', 'ldt', 'synchronisation', 'lassa virus', 'ma 3', 'rrr', 'rbbb', 'iui', 'rdi', 'nsr', 'heod', 'block copolymer', 'moment arm', 'explanatory model', 'e3a', 'olp', 'npe', 'lcp', 'invasive candidiasis', 'cbm', 'treadmilling', 'agb', 'cpaf', 'gaw', 'rectopexy', 'carotid bruit', 'dyserythropoiesis', 'post transfusion purpura', 'community health center', 'h5n2', 'burst fracture', 'lsi', 'sud', 'distemper', 'ohtam', 'pa 1', 'bioglass', 'paradoxical motion', 'school refusal', 'hmf', 'microelectronics', 'thd', 'ipi', 'growth function', 'gellan gum', 'nsg', 'membrane oxygenators', 'dhf', 'scalding', 'eis', 'smm', 'extravert', 'censored data', 'ndi', 'liver cell adenoma', 'b rger', 'max b', 'oedipus', 
                'squalus', 'cordemcura', 'fixed dose combination', 'psychological examination', 'feathering', 'rnase m',       
                'hexenal', 'phthalates', 'mbps', 'laminaria', 'contraceptive prevalence', 'raw water', 'nfd', 'rld', 'optic chiasma', 'iscador', 'c5h5', 'samoan', 'lemur', 'peo', 'rvot', 'acifran', 'erector spinae', 's ferritin', 'aerodynamics', 'norverapamil', 'd ff', 'mamp', 'wcs', 'phase imaging', '8h', 'chenodiol', 'limax maximus', 'dkb', 'rosin', 'vtam', 'original research', 'tissue kallikreins', 'risk homeostasis', 'fenthion', 'charles nicolle', 'basal cell adenoma', 'blepharoplast', 'bipedal', 'cystatins', 'decadron', 'llama', 'oye', 'caesarean operation', 'wiener', 'prosthodontist', 'rse', 'nsts', 't reflex', 'dtf', 'balt', 'sxr', 'pine vole', 'creme', 'underdeveloped country', 'asor', 'd arabinose', 'cognin', 'socialist', 'hfo', 'autogeny', 'bendiocarb', 'extractable nuclear antigen', 'trillion', 'pre flight', 'otolith organ', 'hyperintensity', 'gpd', 'afro american', 'monochloramine', 'pila', 'ferric hydroxide', 'khs', 'nva', 'tritc', 'ft 1', 'gnrf', 'encephalitozoon cuniculi', 'sbr', 'cuticulin', 'peritoneal membrane', 'interorbital', 'thialysine', 'auh', 'raas', 'actinolite', 'sa 11', 'phosphophoryn', 'electro mechanical', 'neophobia', 'steroid sulphatase', 'papc', 'benzoyl chloride', 'tfs', 'rab', 'uroflow', 'phmb', 'oesophageal cancer', 'dna profile', 'gnrh analogue', 'pivampicillin', 'pivmecillinam', 'zuclopenthixol', 'kme', 'kyna', 'metolazone', 'hrc', 'jargon', 'yst', 'matrix granule', 'cameron', 'tbt', 'gyrb', 'agd', 'gardos', 'polyamine oxidase', 'bufo arenarum', 'mphi', 'benzamil', 'putp', 'didelphis marsupialis', 'class 5', 'brain size', 'tcbs', 'mnb', 'ccbs', 'leh', 'hydroxypyruvate', 'broadcast', 'qk', 'pinealocyte', 'gef', 'gta', 'gnt', 'asparagine synthetase', 'd glycerate', 'thrombospondins', 'enameloid', 'phen', 'antheraxanthin', 'corolla', 'ethephon', 'k2cro4', 'mnase', 'tripamide', 'cantonal', 'mmmf', 'toothpick', 'video game', 'basilar papilla', 'vvc', 'hemidesmosome', 'field of research', 'human finger', 'extra uterine pregnancy', 'detrusor sphincter dyssynergia', 'dexamethasone acetate', 'tbn', 'nonprofit hospital', 'lhm', 'ciladopa', 'ssr', 'anterior border', 'midline granuloma', 'speechreading', 'afterdrop', 'paramesonephric duct', 'parkinsonian tremor', 'dlpfc', 'ljm', 'crocodilian', 'vtg', 'sprinkle', 'hexoprenaline', 'vestibular neuronitis', 'npu', 'fek', 'merc', 'trimetaphosphatase', 'mitral atresia', 'cvid', 'v2o5', 'circulatory shock', 'cbb', 'lvpa', 'dandy walker malformation', 'aloe', 'licorice', 'bifemelane', 'pcdds', 'functional value', 'sa7', 'garlic oil', 'kgy', 'swimmeret', 'bordeaux', 'ivdu', 'dorsal motor nucleus', 'mpeg', 'henle', 'ecl cell', 'xl', 'sul', 'tetragastrin', 'bpmc', 'interconnect', 'cladosporium herbarum', '2 ohe1', 'eep', 'fluspirilene', 'imaginal', 'levator ani muscle', 'ba 3', 'on2', 'rcap', 'aminopenicillin', 'deflazacort', 'alfa', 'needle holder', 'oversized', 'subcohort', 'flexural strength', 'citrate cycle', 'pfo', 'leuprolide', 'mauthner neuron', 'mid gut', 'tdn', 'roa', 'rubpc o', 'lt4', 'capsomer', 'dbm', 'bhv', 'musa', 'ccx', 'dci', 'gre', 'ccw', 'dlg', 'silvestris', 'maxillary process', '1 5ag', 'wen', 'quis', 'svp', 'childhood leukaemia', 'profilaggrin', 'circaseptan', 'mae', 'kif', 'protracta', 'dcii', 'gpx', 'genetic sequence', 'ntf', 'power stroke', 'bdx', 'gsk 3', 'csk', 'dga', 'platypus', 'monotreme', 'fadl', 'lcfa', 'fadd', 'ak2', 'seta', 'statocyst', 'cheetah', 'ant nest', 'heliconius', 'levator scapula', 'sdw', 'lactoglobulin', 'phys ther', 'aaav', 'statistical noise', 'concept of self', 'betaf', 'bsl', 'phytosiderophores', 'fluridone', 'red 9', 'naturalist', 'clinostat', 'statoliths', 'plasmodesmata', 'cwf', 'rlc', 'pmrs', 'crystalluria', 'trimebutine', 'chondrolysis', 'itpa', 'ico', 'phenylketonuric', 'buffy layer', 'urticarial vasculitis', 'scheimpflug', 'tpl', 'tammar wallaby', 'triturus cristatus', 'iga nephritis', 'esrf', 'megace', 'lig', 'child and adolescent psychiatry', 'nourseothricin', 'amicar', 'color constancy', 'spectral reflectance', 'dnic', 'neurogenetic', 'dtz', 'alpha ketoacid', 'rae', 'organic mercury', 'k2cr2o7', 'upper oesophageal sphincter', 'nhis', 'hoi', 'ccws', 'citellus', 'nrp', 'tapetum', 'edt', 'eib', 'oas', 'horopter', 'circus movement tachycardia', 'positional vertigo', 'asds', 'neb', 'pisiform', 'bashkirs', 'drug sale', 'sulprostone', 'hbg', 'projective identification', '18f fdg', 'leucogenenol', 'gpr', 'pinaverium bromide', 'ivda', 'autobac', 'papm', 'unfermented', 'lcg', 'snx', 'pdms', 'rheumatic carditis', 'lmf', 'sulphadoxine', 'mada', 'pronephric duct', 'rock wool', 'pkl', 'dmnl', 'wysn', 'hedp', 'noscapine', 'scup', 'vhdl', 'reticulopodial', 'july 23', 'eup', 'neurotoxicology', 'lundby', 'ftorafur', 'p5c', 'biorheology', 'caiman', 'vmb', 'tmx', 'dysequilibrium', 'aprtase', 'triac', 'endralazine', 'vlt', 'sco', 'gaca', '1 decanol', 'mercury chloride', 'cnn', 'sabbatical', 'cranial neural crest', 'acoustic radiation', 'antineoplaston', 'calcium dobesilate', 'fiau', 'nfts', 'pmh', 'srca', 'ethchlorvynol', 'infundibular recess', 'p ser', 'shogaol', 'dban', 'kbro3', 'estradiol enantate', 'copra', 'response cost', 'expeller', 'fnt', 'afterdepolarizations', 'rtz', 'kkns', 'druj', 'benzotrichloride', 'btc', 'wet suit', 'd methamphetamine', 'lpla', 'brain inflammation', 'eads', 'ippa', 'etz', 'pneumogram', 'electrorotation', 'uq', 'afterimage', 'carbosulfan', 'fenticonazole', 'glutaurine', 'dpe', 'prednicarbate', 'lbf', 'paam', 'navelbine', 'rociverine', 'huk', 'fns', 'vu', 'ptilocercus', 'meprin', 'hgh deficiency', 'dwp', 'capozide', 'amae', 'grx', 'rvrr', 'upoc', 'hcq', 'dmdr', 'fenoverine', 'alacepril', 'terlipressin', 'fsg', 'aira', 'unit n', 'sdif', 'numa', 'supt', 'somatrem', 'hir', 'tpk', 'cfn', 'fph', 'bgs', 'lacrisert', 'lupin seed', 'ifos', 'gangliectomy', 'th 3', 'dnl', 'cysb', '3t', 'paii', 'psap', 'hapv', 'biogenic silica', 'acei', 'utmb', 'fhv 1', 'hbn', 'berodual', 'ef 3', 'leukosialin', 'sertraline', 'diesel fuel', 'cdc28', 'dienogest', 'apoa4', 'ermf', 'thpp', 'gvd', 'k lactis', 'queuine', 'topa', 'computer hardware', 'doi', 'esh', 'tipc', 'sn3', 'wsa', 'igcc', 'bl4', 'fpe', 'sultopride', 'afsb', 'nmpc', 'dienochlor', 'mesoglea', 'jp 5', 'okadaic acid', 'sey', 'otocyst', 'ltg', 'ssq', 'respiratory pharmacology', 'pdgfc', 'cpha', 'stbm']
    for i in range(len(topics)-1):
        yield topics[i]