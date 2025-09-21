from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.pieg import PIEGAgent
from algorithms.drqpretrain import DrQV2Agent
from algorithms.drqattention import DrQAtenAgent

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
	'pieg': PIEGAgent,
 	"drqpretrain": DrQV2Agent,
	"drqattention":DrQAtenAgent
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
