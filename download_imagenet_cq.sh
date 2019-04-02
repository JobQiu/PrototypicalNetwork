#!/usr/bin/env bash

#declare -a trainclasses=("n04509417" "n07697537" "n03924679" "n02113712" "n02089867" "n02108551" "n03888605" "n04435653" "n02074367" "n03017168" "n02823428" "n03337140" "n02687172" "n02108089" "n13133613" "n01704323" "n02165456" "n01532829" "n02108915" "n01910747" "n03220513" "n03047690" "n02457408" "n01843383" "n02747177" "n01749939" "n13054560" "n03207743" "n02105505" "n02101006" "n09246464" "n04251144" "n04389033" "n03347037" "n04067472" "n01558993" "n07584110" "n03062245" "n04515003" "n03998194" "n04258138" "n06794110" "n02966193" "n02795169" "n03476684" "n03400231" "n01770081" "n02120079" "n03527444" "n07747607" "n04443257" "n04275548" "n04596742" "n02606052" "n04612504" "n04604644" "n04243546" "n03676483" "n03854065" "n02111277" "n04296562" "n03908618" "n02091831" "n03838899")
#declare -a testclasses=("n02110341" "n01930112" "n02219486" "n02443484" "n01981276" "n02129165" "n04522168" "n02099601" "n03775546" "n02110063" "n02116738" "n03146219" "n02871525" "n03127925" "n03544143" "n03272010" "n07613480" "n04146614" "n04418357" "n04149813")
#declare -a trainclasses=("n11807108" "n10043024" "n01623615" "n10340312" "n04045255" "n01670535" "n11972291" "n12345899" "n04951716" "n06793231" "n13214485" "n01977485" "n02137549" "n09638454" "n03752922" "n01553142" "n01514668" "n01746359" "n03447447" "n01739094" "n02288789" "n04474035" "n09924195" "n07745046" "n02110627" "n09861946" "n01796340" "n04242704" "n12034141" "n12196129" "n04496726" "n13874073" "n04381073" "n01958531" "n07859583" "n03217653" "n02187150" "n07828642" "n03064350" "n04228422" "n12237486" "n10683675" "n07932762" "n03073977" "n03320262" "n12493426" "n02413917" "n12269652" "n10520544" "n03300216" "n04034262" "n04404412" "n03499354" "n03546340" "n07826091" "n03821145" "n02011281" "n10071332" "n11619687" "n10185793" "n02877642" "n02670186" "n10274173" "n02230023" "n09770359" "n02898711" "n03794136" "n01674464" "n02246941" "n04506688" "n01602209" "n03355925" "n09686401" "n03694761" "n03701790" "n03397266" "n02138647" "n02701002" "n12683096" "n01597022" "n02930339" "n10129825" "n04103769" "n04584207" "n01499732" "n12044041" "n02630739" "n02170400" "n07564101" "n12793494" "n02947660" "n03034405" "n03653110" "n03713069" "n02836513" "n01519563" "n12469517" "n02713003" "n04114069")
#declare -a testclasses=("n12098524" "n09706029" "n03339643" "n04578801" "n10453184" "n12754981" "n08238463" "n10669991" "n01922303" "n02694966" "n02793495" "n11730933" "n03276179" "n01733466" "n02095889" "n10487363" "n10533874" "n03896526" "n03500389" "n03950899" "n02896074" "n11837351" "n07833951" "n02854630" "n02979516" "n12070583" "n11668117" "n07934373" "n01770393" "n07844786" "n12723062" "n04397860" "n10051861" "n04111962" "n04407686" "n03563200" "n01828556" "n11739365" "n07899899" "n10536416" "n02146371" "n02903126" "n09421951" "n03006626" "n06267145" "n03802393" "n03536761" "n12301445" "n07648913" "n02382750" )

#declare -a trainclasses=("n02795169" "n04404412" "n04251144" "n03476684" "n01910747" "n04584207" "n02504013" "n04509417" "n03838899" "n03062245" "n03207743" "n02701002" "n03124170" "n03527444" "n01770081" "n03337140" "n02111277" "n02113712" "n03017168" "n13133613" "n02108551" "n02115641" "n01749939" "n12345899" "n02823428" "n02114548" "n03530642" "n01784675" "n03676483" "n04258138" "n03888605" "n04243546" "n03355925" "n09246464" "n11904109" "n02165456" "n02277742" "n04141975" "n13054560" "n04275548" "n07747607" "n02687172" "n02108089" "n03400231" "n02120079" "n03347037" "n03075370" "n02137549" "n02898711" "n04443257" "n03447447" "n02165105" "n01796340" "n02096294" "n04496726" "n03924679" "n02101006" "n07735510" "n07730406" "n03452741" "n03047690" "n04435653" "n02966193" "n02747177" "n02200198" "n02325366" "n01909906" "n04596742" "n04417672" "n04515003" "n04389033" "n03220513" )
#declare -a testclasses=("n02793495" "n01770393" "n01981276" "n03146219" "n04418357" "n03544143" "n09421951" "n02099601" "n03272010" "n02219486" "n03127925" "n12301445" "n02116738" "n02129165" "n01930112" "n12754981" "n02110341" "n06267145" "n04522168" "n04146614" "n02443484" "n02871525" )

declare -a trainclasses=("n02795169" "n04404412" "n04251144" "n03476684" "n01910747" "n02504013" "n04509417" "n03062245" "n03207743" "n02701002" "n03124170" "n03527444" "n04254777" "n03337140" "n02111277" "n03017168" "n02108551" "n02115641" "n01749939" "n12345899" "n02823428" "n03530642" "n01784675" "n03676483" "n01963571" "n03888605" "n04243546" "n03355925" "n11943660" "n09246464" "n11904109" "n02165456" "n01773157" "n02277742" "n04141975" "n13054560" "n04275548" "n07747607" "n02687172" "n04552348" "n02108089" "n03400231" "n02120079" "n02137549" "n02898711" "n04443257" "n03447447" "n02165105" "n02096294" "n04496726" "n03924679" "n02101006" "n07730406" "n03047690" "n04435653" "n02966193" "n02747177" "n02200198" "n02325366" "n01909906" "n04596742" "n04515003" "n04389033" "n03220513" )
declare -a testclasses=("n01770393" "n01981276" "n03146219" "n04418357" "n03544143" "n09421951" "n02099601" "n03272010" "n02219486" "n03127925" "n12301445" "n02116738" "n02129165" "n01930112" "n02110341" "n06267145" "n04522168" "n04146614" "n02443484" "n02871525" )
declare -a valclasses=("n02793495" "n04584207" "n12302071" "n03838899" "n01770081" "n02113712" "n13133613" "n02114548" "n04258138" "n03347037" "n03075370" "n12754981" "n01796340" "n07735510" "n03452741" "n04417672" )

token=a305d2c8a6e74485e9ec4b5a9b9fa6df1825ed33
user=congminqiu
DATADIR=data/mini-imagenet

# Create directories
mkdir -p $DATADIR/data
mkdir -p $DATADIR/data/train
mkdir -p $DATADIR/data/test
mkdir -p $DATADIR/data/val

# Download train set
for i in "${trainclasses[@]}"
do
   wget -c --output-document=$DATADIR/data/train/$i.tar "http://www.image-net.org/download/synset?wnid=$i&username=$user&accesskey=$token&release=latest&src=stanford"
   mkdir $DATADIR/data/train/$i
   tar xf $DATADIR/data/train/$i.tar -C $DATADIR/data/train/$i/
done

# Download test set
for j in "${testclasses[@]}"
do
   wget -c --output-document=$DATADIR/data/test/$j.tar "http://www.image-net.org/download/synset?wnid=$j&username=$user&accesskey=$token&release=latest&src=stanford"
   mkdir $DATADIR/data/test/$j
   tar xf $DATADIR/data/test/$j.tar -C $DATADIR/data/test/$j/
done


# Download val set
for j in "${valclasses[@]}"
do
   wget -c --output-document=$DATADIR/data/val/$j.tar "http://www.image-net.org/download/synset?wnid=$j&username=$user&accesskey=$token&release=latest&src=stanford"
   mkdir $DATADIR/data/val/$j
   tar xf $DATADIR/data/val/$j.tar -C $DATADIR/data/val/$j/
done