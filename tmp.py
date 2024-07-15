pdb_id = 'A0A2Z2J8L1'

from ost.io import LoadPDB
from ost.mol.alg import (CleanlDDTReferences, PreparelDDTGlobalRDMap, lDDTSettings, CheckStructure, LocalDistDiffTest, GetlDDTPerResidueStats,PrintlDDTPerResidueStats, ResidueNamesMatch)
from ost.io import ReadStereoChemicalPropsFile
model_path = f"/home/tlp5359/projects/CC_Sequencing_LLM/BCB/pdbs/UniRef100_{pdb_id}/selected_prediction.pdb"
reference_path = f"/home/tlp5359/projects/CC_Sequencing_LLM/BCB/pdbs/AF-{pdb_id}-F1-model_v4.pdb"
structural_checks = True
bond_tolerance = 12
angle_tolerance = 12
cutoffs = [0.5, 1.0, 2.0, 4.0]
model = LoadPDB(model_path)
model_view = model.GetChainList()[0].Select("peptide=true")
references = [LoadPDB(reference_path).CreateFullView()]
settings = lDDTSettings()
settings.PrintParameters()
CleanlDDTReferences(references)
rdmap = PreparelDDTGlobalRDMap(references,
                            cutoffs=cutoffs,
                            sequence_separation=settings.sequence_separation,
                            radius=settings.radius)
if structural_checks:
    stereochemical_parameters = ReadStereoChemicalPropsFile()
    CheckStructure(ent=model_view,
                bond_table=stereochemical_parameters.bond_table,
                angle_table=stereochemical_parameters.angle_table,
                nonbonded_table=stereochemical_parameters.nonbonded_table,
                bond_tolerance=bond_tolerance,
                angle_tolerance=angle_tolerance)
is_cons = ResidueNamesMatch(model_view, references[0], True)
print("Consistency check: ", "OK" if is_cons else "ERROR")
LocalDistDiffTest(model_view,
                references,
                rdmap,
                settings)
# local_scores = GetlDDTPerResidueStats(model_view,
#                                     rdmap,
#                                     structural_checks,
#                                     settings.label)
# PrintlDDTPerResidueStats(local_scores, structural_checks, len(cutoffs))