rm -f /home/genxadmin/TG/data/chrom_files.txt
for chr in $(seq 1 22); do
    fn=ALL.chr${chr}.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz
    full_path=/home/genxadmin/TG/data/raw/ALL.chr${chr}.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz
    echo "Downloading "$fn
    rm -f $full_path
    wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20181203_biallelic_SNV/${fn} -O ${full_path}
    wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20181203_biallelic_SNV/${fn}.tbi -O ${full_path}.tbi
    echo $full_path >> /home/genxadmin/TG/data/chrom_files.txt
done


echo "Merging..."
rm -f /home/genxadmin/TG/data/tg.vcf.gz
/home/genxadmin/opt/bcftools/bin/bcftools concat --file-list /home/genxadmin/TG/data/chrom_files.txt -Oz -o /home/genxadmin/TG/data/tg.vcf.gz

echo "Converting with PLINK..."
rm -f /home/genxadmin/TG/data/tg.bim
rm -f /home/genxadmin/TG/data/tg.bed
rm -f /home/genxadmin/TG/data/tg.fam
/home/genxadmin/opt/plink2 --vcf /home/genxadmin/TG/data/tg.vcf.gz --make-bed --out /home/genxadmin/TG/data/tg
